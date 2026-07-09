"""Reranking via Cohere (LiteLLM) or LLM fallback — always scores child chunks."""

from __future__ import annotations

import json

import httpx

from app.core.config import Settings
from app.core.cost_tracker import CostTracker
from app.core.logging import get_logger
from services.llm import LLMService

log = get_logger(__name__)


def _passage_text(candidate: dict, max_len: int = 800) -> str:
    """Use the matched child chunk for scoring; parent context is not a rank signal."""
    return (candidate.get("text") or "")[:max_len]


def _blend_scores(candidates: list[dict], rerank_scores: list[float]) -> list[dict]:
    """Combine reranker output with retrieval fusion scores for stable ordering."""
    if not candidates:
        return []

    fusion = [float(c.get("fusion_score", c.get("score", 0.0))) for c in candidates]
    max_fusion = max(fusion) or 1.0
    max_rerank = max(rerank_scores) or 1.0
    min_rerank = min(rerank_scores)
    flat_rerank = max_rerank == min_rerank

    scored: list[dict] = []
    for cand, rerank, fuse in zip(candidates, rerank_scores, fusion):
        entry = cand.copy()
        entry["retrieval_score"] = float(cand.get("fusion_score", cand.get("score", 0.0)))
        entry["rerank_score"] = float(rerank)
        norm_rerank = rerank / max_rerank if max_rerank else 0.0
        norm_fusion = fuse / max_fusion if max_fusion else 0.0
        if flat_rerank:
            entry["score"] = norm_fusion
        else:
            entry["score"] = round(0.65 * norm_rerank + 0.35 * norm_fusion, 6)
        scored.append(entry)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


class RerankerService:
    def __init__(
        self,
        settings: Settings,
        llm: LLMService,
        cost_tracker: CostTracker | None = None,
    ):
        self._settings = settings
        self._llm = llm
        self._cost = cost_tracker

    async def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        if not candidates:
            return []

        pool = []
        for c in candidates[: self._settings.rerank_top_n]:
            entry = c.copy()
            entry["fusion_score"] = float(c.get("score", 0.0))
            pool.append(entry)

        if self._cohere_available():
            try:
                return await self._cohere_rerank(query, pool, top_k)
            except Exception as exc:
                log.warning("cohere_rerank_failed", error=str(exc), fallback="llm")

        if self._settings.enable_rerank:
            return await self._llm_rerank(query, pool, top_k)

        return pool[:top_k]

    def _cohere_available(self) -> bool:
        return self._settings.enable_cohere_rerank and bool(
            self._settings.cohere_api_key.strip()
        )

    async def _cohere_rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        documents = [_passage_text(c, 4096) for c in candidates]

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self._settings.litellm_base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self._settings.litellm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._settings.rerank_model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if self._cost:
            self._cost.log_litellm(
                job_id=None,
                document_id=None,
                model=self._settings.rerank_model,
                operation="cohere_rerank",
                input_tokens=len(query.split()),
                output_tokens=0,
                extra={"documents": len(documents), "top_k": top_k},
            )

        results = data.get("results", data.get("data", []))
        rerank_scores = [0.0] * len(candidates)
        for item in results:
            idx = item.get("index", item.get("document_index", 0))
            if 0 <= idx < len(candidates):
                rerank_scores[idx] = float(
                    item.get("relevance_score", item.get("score", 0.0))
                )

        ranked = _blend_scores(candidates, rerank_scores)
        for entry in ranked[:top_k]:
            entry["source"] = "cohere_rerank"
        return ranked[:top_k]

    async def _llm_rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        passages = [f"[{i}] {_passage_text(c)}" for i, c in enumerate(candidates)]

        prompt = f"""Score each passage for relevance to the query.
Use a float from 0.0 (irrelevant) to 10.0 (perfect match).
Penalize passages that do not contain information needed to answer the query.
Return ONLY a JSON array sorted by score descending:
[{{"index": int, "score": float}}, ...]

Query: {query}

Passages:
{chr(10).join(passages)}"""

        try:
            response, _ = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                operation="llm_rerank_fallback",
                temperature=0.0,
            )
            raw = response.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores_data = json.loads(raw)

            rerank_scores = [0.0] * len(candidates)
            for item in scores_data:
                idx = int(item["index"])
                if 0 <= idx < len(candidates):
                    rerank_scores[idx] = float(item["score"])

            ranked = _blend_scores(candidates, rerank_scores)
            for entry in ranked[:top_k]:
                entry["source"] = "llm_rerank"
            return ranked[:top_k]
        except Exception:
            return candidates[:top_k]
