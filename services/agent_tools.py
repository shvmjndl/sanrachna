"""Agent tool definitions and executors."""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import select

from app.core.config import Settings
from app.db.models import Document
from app.schemas.knowledge import Entity, SearchResult
from pipeline.indexing import HybridIndexer
from services.compare import CompareService
from services.compliance import check_document_compliance
from services.contradiction import ContradictionService
from services.entity_resolution import EntityResolver
from services.graph_service import GraphService
from services.llm import LLMService
from services.translate import TranslateService

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Hybrid search (vector + BM25 + Cohere rerank). Primary tool for factual questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_lookup",
            "description": "Find entities and relationships in knowledge graph. Use for org/person/vendor connections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_query": {"type": "string"},
                },
                "required": ["entity_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "entity_search",
            "description": "Search canonical entities across corpus (GSTIN, PAN, ORG, PERSON). Returns cross-document matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "entity_type": {
                        "type": "string",
                        "description": "Optional filter: GSTIN, PAN, IFSC, ORG, PERSON, MONEY",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "table_query",
            "description": "Search structured tables extracted from documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all documents in the corpus with metadata (filename, language, entity count).",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_summary",
            "description": "Get summary of a specific document: entities, tables, chunks, markdown excerpt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                },
                "required": ["document_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compliance_check",
            "description": "Validate GSTIN, PAN, IFSC, Aadhaar formats in a document or across corpus. Use for audit/compliance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Optional — check one doc, else whole corpus"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text between Indian languages using Sarvam. Use when query language differs from document language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_language": {
                        "type": "string",
                        "description": "BCP-47 code: hi-IN, en-IN, ta-IN, etc.",
                        "default": "en-IN",
                    },
                    "source_language": {"type": "string", "default": "auto"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_documents",
            "description": "Compare entities and tables between two documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id_a": {"type": "string"},
                    "document_id_b": {"type": "string"},
                },
                "required": ["document_id_a", "document_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_evidence",
            "description": "Summarize gathered search results into a concise brief. Use after collecting evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {"type": "string", "description": "What aspect to focus the summary on"},
                },
                "required": ["focus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a numeric expression from gathered evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_traverse",
            "description": "Multi-hop knowledge graph traversal from a starting entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_entity_query": {"type": "string"},
                    "max_hops": {"type": "integer", "default": 3},
                },
                "required": ["start_entity_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_contradictions",
            "description": "Find conflicting values between two documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id_a": {"type": "string"},
                    "document_id_b": {"type": "string"},
                },
                "required": ["document_id_a", "document_id_b"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are Sanrachna, an expert AI for Indian-language document intelligence.

You have 13 tools. Always gather evidence with tools before answering. Cite chunk IDs.

Strategy:
- Factual questions → search_documents
- Tabular or row-structured facts → table_query + search_documents
- Who/what/vendor/org → graph_lookup + entity_search
- Audit/compliance → compliance_check
- Cross-lingual → translate_text then search_documents
- Multi-doc analysis → list_documents, compare_documents
- Complex questions → use multiple tools across steps

Respond in the user's language when possible."""


class ToolExecutor:
    def __init__(
        self,
        settings: Settings,
        llm: LLMService,
        indexer: HybridIndexer,
        graph: GraphService,
        entity_resolver: EntityResolver,
        session_factory,
    ):
        self._settings = settings
        self._llm = llm
        self._indexer = indexer
        self._graph = graph
        self._entities = entity_resolver
        self._session = session_factory
        self._translate = (
            TranslateService(settings.sarvam_api_key) if settings.sarvam_api_key else None
        )
        self._document_id: str | None = None
        self._corpus_id: str | None = None
        self._citations: list[SearchResult] = []
        self._found_entities: list[Entity] = []
        self._found_tables: list[dict] = []

    def set_context(self, document_id: str | None, corpus_id: str | None) -> None:
        self._document_id = document_id
        self._corpus_id = corpus_id

    @property
    def graph(self) -> GraphService:
        return self._graph

    @property
    def citations(self) -> list[SearchResult]:
        return self._citations

    @property
    def found_entities(self) -> list[Entity]:
        return self._found_entities

    @property
    def found_tables(self) -> list[dict]:
        return self._found_tables

    async def execute(self, name: str, args: dict, top_k: int) -> Any:
        handler = getattr(self, f"_tool_{name}", None)
        if not handler:
            return {"error": f"Unknown tool: {name}"}
        return await handler(args, top_k)

    async def _tool_search_documents(self, args: dict, top_k: int) -> Any:
        results = await self._indexer.search(
            query=args["query"],
            top_k=args.get("top_k", top_k),
            document_id=self._document_id,
            corpus_id=self._corpus_id,
            hybrid=True,
            rerank=True,
            expand_parent=False,
        )
        self._citations.extend(results)
        return [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text[:500],
                "score": r.score,
                "source": r.source,
            }
            for r in results
        ]

    async def _tool_graph_lookup(self, args: dict, top_k: int) -> Any:
        entities, rels, graph = self._graph.lookup_entities(
            args["entity_query"],
            corpus_id=self._corpus_id,
            document_id=self._document_id,
        )
        self._found_entities.extend(entities)
        return {
            "entities": [e.model_dump() for e in entities],
            "relationships": [r.model_dump() for r in rels],
            "graph_edges": [e.model_dump() for e in graph.edges],
        }

    async def _tool_entity_search(self, args: dict, top_k: int) -> Any:
        matches = self._entities.search_entities(
            args["query"], corpus_id=self._corpus_id, limit=top_k
        )
        if args.get("entity_type"):
            et = args["entity_type"].upper()
            matches = [m for m in matches if m["type"].upper() == et]
        return matches

    async def _tool_table_query(self, args: dict, top_k: int) -> Any:
        tables = self._graph.search_tables(
            args["query"],
            corpus_id=self._corpus_id,
            document_id=self._document_id,
        )
        self._found_tables.extend(tables)
        return tables

    async def _tool_list_documents(self, args: dict, top_k: int) -> Any:
        with self._session() as session:
            stmt = select(Document)
            if self._corpus_id:
                stmt = stmt.where(Document.corpus_id == self._corpus_id)
            elif self._document_id:
                stmt = stmt.where(Document.id == self._document_id)
            docs = session.execute(stmt.limit(50)).scalars().all()

        return [
            {
                "document_id": d.id,
                "filename": d.filename,
                "language": d.language,
                "corpus_id": d.corpus_id,
                "entity_count": len((d.knowledge_object or {}).get("entities", [])),
                "table_count": len((d.knowledge_object or {}).get("tables", [])),
                "chunk_count": len((d.knowledge_object or {}).get("chunks", [])),
            }
            for d in docs
        ]

    async def _tool_get_document_summary(self, args: dict, top_k: int) -> Any:
        doc_id = args["document_id"]
        with self._session() as session:
            doc = session.execute(
                select(Document).where(Document.id == doc_id)
            ).scalar_one_or_none()

        if not doc or not doc.knowledge_object:
            return {"error": f"Document {doc_id} not found"}

        ko = doc.knowledge_object
        return {
            "document_id": doc_id,
            "filename": doc.filename,
            "language": doc.language,
            "entity_count": len(ko.get("entities", [])),
            "table_count": len(ko.get("tables", [])),
            "relationship_count": len(ko.get("relationships", [])),
            "entities": ko.get("entities", [])[:10],
            "tables": [
                {"id": t.get("id"), "caption": t.get("caption"), "headers": t.get("headers")}
                for t in ko.get("tables", [])[:5]
            ],
            "markdown_excerpt": ko.get("markdown", "")[:1500],
        }

    async def _tool_compliance_check(self, args: dict, top_k: int) -> Any:
        doc_id = args.get("document_id") or self._document_id
        with self._session() as session:
            stmt = select(Document)
            if doc_id:
                stmt = stmt.where(Document.id == doc_id)
            elif self._corpus_id:
                stmt = stmt.where(Document.corpus_id == self._corpus_id)
            docs = session.execute(stmt).scalars().all()

        all_checks = []
        for doc in docs:
            if not doc.knowledge_object:
                continue
            entities = doc.knowledge_object.get("entities", [])
            result = check_document_compliance(entities)
            all_checks.append(
                {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    **result,
                }
            )

        total_invalid = sum(c["summary"]["invalid"] for c in all_checks)
        return {
            "corpus_compliant": total_invalid == 0,
            "documents_checked": len(all_checks),
            "results": all_checks,
        }

    async def _tool_translate_text(self, args: dict, top_k: int) -> Any:
        if not self._translate:
            return {"error": "Translation unavailable — set SARVAM_API_KEY"}
        return self._translate.translate(
            args["text"],
            source_language=args.get("source_language", "auto"),
            target_language=args.get("target_language", "en-IN"),
        )

    async def _tool_compare_documents(self, args: dict, top_k: int) -> Any:
        return CompareService(self._session).compare(args["document_id_a"], args["document_id_b"])

    async def _tool_graph_traverse(self, args: dict, top_k: int) -> Any:
        result = self._graph.traverse(
            args["start_entity_query"],
            max_hops=args.get("max_hops", 3),
            corpus_id=self._corpus_id,
            document_id=self._document_id,
        )
        return {
            "start_entities": [e.model_dump() for e in result.get("start_entities", [])],
            "paths": result.get("paths", []),
            "nodes": [n.model_dump() for n in result.get("nodes", [])],
            "edges": [e.model_dump() for e in result.get("edges", [])],
        }

    async def _tool_detect_contradictions(self, args: dict, top_k: int) -> Any:
        return ContradictionService(self._session).detect_between(
            args["document_id_a"], args["document_id_b"]
        )

    async def _tool_summarize_evidence(self, args: dict, top_k: int) -> Any:
        if not self._citations:
            return {"error": "No evidence gathered yet — run search_documents first"}

        evidence = "\n\n".join(
            f"[{c.chunk_id}]: {c.text[:600]}" for c in self._citations[:8]
        )
        prompt = f"""Summarize the following evidence focusing on: {args['focus']}
Be concise, factual, cite chunk IDs. Use bullet points.

Evidence:
{evidence}"""

        summary, _ = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            operation="summarize_evidence",
        )
        return {"focus": args["focus"], "summary": summary, "sources": len(self._citations)}

    async def _tool_calculate(self, args: dict, top_k: int) -> Any:
        return _safe_calculate(args["expression"])

    def _load_doc(self, doc_id: str) -> dict | None:
        with self._session() as session:
            doc = session.execute(
                select(Document).where(Document.id == doc_id)
            ).scalar_one_or_none()
        return doc.knowledge_object if doc else None


def _safe_calculate(expression: str) -> dict:
    allowed = re.compile(r"^[\d\s+\-*/().,%₹]+$")
    clean = expression.replace("₹", "").replace(",", "").strip()
    if not allowed.match(clean):
        return {"error": "Invalid expression"}
    try:
        result = eval(clean, {"__builtins__": {}}, {})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}
