"""True hybrid search: dense + BM25 + RRF + Cohere rerank."""

from __future__ import annotations

import uuid
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import Settings
from app.core.cost_tracker import CostTracker
from app.schemas.knowledge import BBox, Chunk, SearchResult
from services.bm25 import BM25Indexer
from services.llm import LLMService
from services.reranker import RerankerService


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results, 1):
            cid = item["chunk_id"]
            scores[cid] += 1.0 / (k + rank)
            if cid not in items:
                items[cid] = item
            else:
                items[cid] = _merge_hit(items[cid], item)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged = []
    for cid, score in ranked:
        entry = {**items[cid], "score": score, "fusion_score": score}
        merged.append(entry)
    return merged


def _merge_hit(primary: dict, secondary: dict) -> dict:
    merged = primary.copy()
    for key in ("bbox", "confidence", "page", "section", "parent_text"):
        if not merged.get(key) and secondary.get(key):
            merged[key] = secondary[key]
    if secondary.get("confidence", 0) > merged.get("confidence", 0):
        merged["confidence"] = secondary["confidence"]
    return merged


class HybridIndexer:
    def __init__(
        self,
        settings: Settings,
        llm: LLMService,
        session_factory=None,
        cost_tracker: CostTracker | None = None,
    ):
        self._settings = settings
        self._llm = llm
        self._client = QdrantClient(url=settings.qdrant_url)
        self._collection = settings.qdrant_collection
        self._bm25 = BM25Indexer(session_factory) if session_factory else None
        self._reranker = RerankerService(settings, llm, cost_tracker)
        self._ensure_collection()

    def _ensure_collection(self, dimensions: int = 1536) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            )

    async def index_chunks(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        corpus_id: str | None = None,
        job_id: str | None = None,
    ) -> int:
        if not chunks or not embeddings:
            return 0

        if len(embeddings[0]) != 1536:
            self._ensure_collection(len(embeddings[0]))

        points = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": document_id,
                        "corpus_id": corpus_id,
                        "text": chunk.text,
                        "parent_text": chunk.parent_text,
                        "page": chunk.page,
                        "section": chunk.section,
                        "token_count": chunk.token_count,
                        "confidence": chunk.confidence,
                        "bbox": chunk.bbox.model_dump() if chunk.bbox else None,
                    },
                )
            )

        self._client.upsert(collection_name=self._collection, points=points)

        if self._bm25:
            self._bm25.index_chunks(chunks, document_id, corpus_id)

        return len(points)

    async def _dense_search(
        self,
        query: str,
        top_k: int,
        document_id: str | None,
        corpus_id: str | None,
    ) -> list[dict]:
        query_embedding, _ = await self._llm.embed(query, operation="search_query")

        filters = []
        if document_id:
            filters.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
        if corpus_id:
            filters.append(FieldCondition(key="corpus_id", match=MatchValue(value=corpus_id)))

        query_filter = Filter(must=filters) if filters else None

        response = self._client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        results = response.points

        return [
            {
                "chunk_id": (hit.payload or {}).get("chunk_id", str(hit.id)),
                "document_id": (hit.payload or {}).get("document_id", ""),
                "text": (hit.payload or {}).get("text", ""),
                "parent_text": (hit.payload or {}).get("parent_text"),
                "page": (hit.payload or {}).get("page"),
                "section": (hit.payload or {}).get("section"),
                "confidence": (hit.payload or {}).get("confidence", 1.0),
                "bbox": (hit.payload or {}).get("bbox"),
                "score": hit.score,
                "source": "dense",
            }
            for hit in results
        ]

    async def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: str | None = None,
        corpus_id: str | None = None,
        hybrid: bool = True,
        rerank: bool = True,
        expand_parent: bool = False,
    ) -> list[SearchResult]:
        fetch_k = max(top_k * 3, self._settings.rerank_top_n) if hybrid else top_k

        dense_results = await self._dense_search(query, fetch_k, document_id, corpus_id)

        if hybrid and self._bm25:
            bm25_results = self._bm25.search(query, fetch_k, document_id, corpus_id)
            fused = reciprocal_rank_fusion([dense_results, bm25_results])
        else:
            fused = dense_results

        if rerank and fused:
            fused = await self._reranker.rerank(query, fused, top_k)
        else:
            fused = fused[:top_k]

        search_results = []
        for hit in fused:
            display_text = hit["text"]
            if expand_parent and hit.get("parent_text"):
                display_text = hit["parent_text"]

            bbox = None
            if hit.get("bbox"):
                try:
                    bbox = BBox(**hit["bbox"])
                except Exception:
                    bbox = None

            retrieval = hit.get("retrieval_score", hit.get("score"))
            rerank = hit.get("rerank_score")

            search_results.append(
                SearchResult(
                    chunk_id=hit["chunk_id"],
                    document_id=hit["document_id"],
                    text=display_text,
                    score=hit["score"],
                    page=hit.get("page"),
                    section=hit.get("section"),
                    parent_text=hit.get("parent_text"),
                    source=hit.get("source", "hybrid"),
                    bbox=bbox,
                    confidence=float(hit.get("confidence", 1.0)),
                    retrieval_score=float(retrieval) if retrieval is not None else None,
                    rerank_score=float(rerank) if rerank is not None else None,
                )
            )

        return search_results

    async def embed_chunks(
        self,
        chunks: list[Chunk],
        job_id: str | None = None,
        document_id: str | None = None,
    ) -> list[list[float]]:
        texts = [c.text for c in chunks]
        if not texts:
            return []
        embeddings, _ = await self._llm.embed_batch(
            texts, operation="chunk_embedding", job_id=job_id, document_id=document_id
        )
        return embeddings

    def delete_document(self, document_id: str) -> int:
        from qdrant_client.models import Filter, FilterSelector, FieldCondition, MatchValue

        result = self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        return bool(result)
