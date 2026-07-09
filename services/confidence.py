"""Evidence confidence scoring for search and agent responses."""

from __future__ import annotations

from app.schemas.knowledge import EvidenceAssessment, SearchResult


def _confidence_level(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    if score >= 0.3:
        return "low"
    return "very_low"


def assess_evidence(
    citations: list[SearchResult],
    *,
    tables: list[dict] | None = None,
    answer: str | None = None,
    quality_score: float | None = None,
) -> EvidenceAssessment:
    """Fast heuristic confidence — no extra LLM call."""
    if not citations:
        return EvidenceAssessment(
            confidence=0.0,
            level="very_low",
            reasons=["No retrieval hits"],
            warnings=["No indexed evidence matched the query"],
            citation_count=0,
            table_support=bool(tables),
        )

    top_scores = [c.score for c in citations[:3]]
    avg_top = sum(top_scores) / len(top_scores)
    chunk_conf = sum(c.confidence for c in citations[:5]) / min(5, len(citations))
    bbox_hits = sum(1 for c in citations if c.bbox)
    bbox_ratio = bbox_hits / len(citations)

    score = 0.45 * min(1.0, avg_top) + 0.30 * chunk_conf + 0.15 * min(1.0, len(citations) / 3)
    if tables:
        score += 0.10
    if bbox_ratio >= 0.5:
        score += 0.05

    reasons = [
        f"{len(citations)} citation(s) retrieved",
        f"top score {top_scores[0]:.3f}",
    ]
    if tables:
        reasons.append("structured table rows available")
    if bbox_ratio > 0:
        reasons.append(f"{bbox_hits} grounded bbox citation(s)")

    warnings: list[str] = []
    if quality_score is not None and quality_score < 0.45:
        warnings.append(f"document quality score is low ({quality_score:.2f})")
        score *= 0.85
    if avg_top < 0.25:
        warnings.append("retrieval scores are weak")
        score *= 0.8
    if answer:
        lower = answer.lower()
        if any(p in lower for p in ("couldn't find", "could not find", "no information", "not sure")):
            warnings.append("answer language indicates uncertainty")
            score *= 0.7

    score = round(max(0.0, min(1.0, score)), 3)
    return EvidenceAssessment(
        confidence=score,
        level=_confidence_level(score),
        reasons=reasons,
        warnings=warnings,
        citation_count=len(citations),
        table_support=bool(tables),
    )
