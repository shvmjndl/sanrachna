"""RAG/agent evaluation harness."""

from __future__ import annotations

from app.core.config import Settings
from app.core.cost_tracker import CostTracker
from app.core.logging import get_logger
from app.schemas.knowledge import (
    EvalCase,
    EvalCaseResult,
    EvalMetrics,
    EvalRequest,
    EvalResponse,
)
from pipeline.indexing import HybridIndexer
from services.agent import DocumentAgent
from services.confidence import assess_evidence
from services.entity_resolution import EntityResolver
from services.graph_service import GraphService
from services.llm import LLMService


class EvalHarness:
    def __init__(self, settings: Settings, session_factory, log):
        self._settings = settings
        self._session = session_factory
        self._log = log

    async def run(self, req: EvalRequest) -> EvalResponse:
        cost_tracker = CostTracker(self._session, self._log)
        llm = LLMService(self._settings, cost_tracker)
        indexer = HybridIndexer(self._settings, llm, self._session, cost_tracker)
        graph = GraphService(self._session)
        entity_resolver = EntityResolver(self._session)
        agent = DocumentAgent(
            settings=self._settings,
            llm=llm,
            indexer=indexer,
            graph=graph,
            entity_resolver=entity_resolver,
            session_factory=self._session,
        )

        results: list[EvalCaseResult] = []
        for case in req.cases:
            results.append(await self._run_case(case, req, agent, indexer))

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        metrics = EvalMetrics(
            total=total,
            passed=passed,
            pass_rate=round(passed / total, 3) if total else 0.0,
            avg_recall_at_k=round(sum(r.recall_at_k for r in results) / total, 3) if total else 0.0,
            avg_confidence=round(sum(r.confidence for r in results) / total, 3) if total else 0.0,
        )
        return EvalResponse(metrics=metrics, results=results)

    async def _run_case(
        self,
        case: EvalCase,
        req: EvalRequest,
        agent: DocumentAgent,
        indexer: HybridIndexer,
    ) -> EvalCaseResult:
        if req.mode == "search":
            hits = await indexer.search(
                query=case.query,
                top_k=req.top_k,
                document_id=case.document_id,
                corpus_id=case.corpus_id,
                hybrid=True,
                rerank=True,
            )
            answer = hits[0].text if hits else ""
            top_chunk_ids = [h.chunk_id for h in hits[: req.top_k]]
            confidence = assess_evidence(hits).confidence
        else:
            response = await agent.run(
                query=case.query,
                document_id=case.document_id,
                corpus_id=case.corpus_id,
                top_k=req.top_k,
                max_steps=6,
            )
            answer = response.answer
            top_chunk_ids = [c.chunk_id for c in response.citations[: req.top_k]]
            confidence = response.evidence.confidence if response.evidence else 0.0

        lower = answer.lower()
        missing = [term for term in case.expected_contains if term.lower() not in lower]
        forbidden = [term for term in case.expected_not_contains if term.lower() in lower]

        recall = 0.0
        if case.expected_chunk_ids:
            expected = set(case.expected_chunk_ids)
            recall = len(expected & set(top_chunk_ids)) / len(expected)

        passed = not missing and not forbidden
        if case.expected_chunk_ids:
            passed = passed and recall >= 0.5

        return EvalCaseResult(
            query=case.query,
            passed=passed,
            answer=answer[:2000],
            top_chunk_ids=top_chunk_ids,
            missing_terms=missing,
            forbidden_terms=forbidden,
            recall_at_k=round(recall, 3),
            confidence=confidence,
        )
