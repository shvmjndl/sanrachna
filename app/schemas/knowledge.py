from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkType(str, Enum):
    PARENT = "parent"
    CHILD = "child"


class BBox(BaseModel):
    """Axis-aligned bounding box in page coordinates (PDF points)."""

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float | None = None
    page_height: float | None = None

    def normalized(self) -> dict[str, float]:
        if not self.page_width or not self.page_height:
            return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}
        return {
            "x0": round(self.x0 / self.page_width, 4),
            "y0": round(self.y0 / self.page_height, 4),
            "x1": round(self.x1 / self.page_width, 4),
            "y1": round(self.y1 / self.page_height, 4),
        }


class ProvenanceInfo(BaseModel):
    source: str = "unknown"  # local | sarvam | convert
    block_id: str | None = None
    layout_tag: str | None = None


class SourceInfo(BaseModel):
    filename: str
    path: str | None = None
    mime_type: str | None = None
    language: str = "hi-IN"
    detected_type: str | None = None
    detected_languages: list[str] = Field(default_factory=list)
    document_family: str | None = None
    family_signals: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    text: str
    page: int | None = None
    section: str | None = None
    token_count: int = 0
    embedding_ref: str | None = None
    parent_id: str | None = None
    parent_text: str | None = None
    chunk_type: ChunkType = ChunkType.CHILD
    bbox: BBox | None = None
    provenance: ProvenanceInfo | None = None
    confidence: float = 1.0


class Table(BaseModel):
    id: str
    caption: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    page: int | None = None
    page_end: int | None = None
    raw_markdown: str | None = None


class ImageAsset(BaseModel):
    id: str
    path: str | None = None
    caption: str | None = None
    ocr_text: str | None = None
    page: int | None = None


class Entity(BaseModel):
    id: str
    type: str
    text: str
    normalized: str | None = None
    confidence: float = 1.0
    source_chunk: str | None = None
    script: str | None = None
    canonical_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    subject: str
    predicate: str
    object: str
    evidence: str | None = None
    confidence: float = 0.0


class GraphNode(BaseModel):
    id: str
    label: str
    type: str


class GraphEdge(BaseModel):
    source: str
    target: str
    predicate: str
    confidence: float = 0.0


class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class EmbeddingsInfo(BaseModel):
    model: str
    dimensions: int = 0
    chunk_count: int = 0


class ProcessingMetadata(BaseModel):
    pages: int = 0
    processing_time_ms: int = 0
    pipeline_version: str = "0.2.0"
    sarvam_job_ids: list[str] = Field(default_factory=list)
    quality_score: float | None = None
    language: str = "hi-IN"
    detected_languages: list[str] = Field(default_factory=list)


class KnowledgeObject(BaseModel):
    document_id: str
    source: SourceInfo
    markdown: str = ""
    chunks: list[Chunk] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    images: list[ImageAsset] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    knowledge_graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)
    embeddings: EmbeddingsInfo | None = None
    metadata: ProcessingMetadata = Field(default_factory=ProcessingMetadata)


class CorpusSummary(BaseModel):
    corpus_id: str
    documents: int = 0
    document_ids: list[str] = Field(default_factory=list)
    entities: int = 0
    relationships: int = 0
    status: JobStatus = JobStatus.PENDING
    created_at: datetime | None = None


class IngestResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str = "Job queued"


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    documents_total: int = 0
    documents_processed: int = 0
    error: str | None = None
    result: CorpusSummary | None = None


class IngestJobSummary(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    progress: float = 0.0
    documents_total: int = 0
    documents_processed: int = 0
    error: str | None = None
    corpus_id: str | None = None
    created_at: datetime | None = None


class IngestJobListResponse(BaseModel):
    jobs: list[IngestJobSummary]
    total: int
    active_count: int


class JobCancelResponse(BaseModel):
    job_id: str
    cancelled: bool
    message: str
    removed_from_queue: int = 0


class JobCancelAllResponse(BaseModel):
    cancelled_count: int
    job_ids: list[str] = Field(default_factory=list)
    removed_from_queue: int = 0


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    document_id: str | None = None
    corpus_id: str | None = None
    hybrid: bool = True
    rerank: bool = True
    expand_parent: bool = False


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    page: int | None = None
    section: str | None = None
    parent_text: str | None = None
    source: str = "hybrid"
    bbox: BBox | None = None
    confidence: float = 1.0
    retrieval_score: float | None = None
    rerank_score: float | None = None


class EvidenceAssessment(BaseModel):
    confidence: float
    level: str
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    citation_count: int = 0
    table_support: bool = False


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int
    graph_context: str | None = None
    evidence: EvidenceAssessment | None = None


class AgentQueryRequest(BaseModel):
    query: str
    document_id: str | None = None
    corpus_id: str | None = None
    top_k: int = 5
    max_steps: int = 5


class AgentStep(BaseModel):
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""


class AgentQueryResponse(BaseModel):
    answer: str
    citations: list[SearchResult] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    graph: KnowledgeGraph | None = None
    tables: list[dict] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    cost_usd: float | None = None
    evidence: EvidenceAssessment | None = None
    scope: dict[str, Any] | None = None


class FieldSpec(BaseModel):
    name: str
    description: str
    type: str = "string"


class ExtractRequest(BaseModel):
    document_id: str | None = None
    corpus_id: str | None = None
    template_id: str | None = None
    fields: list[FieldSpec] = Field(default_factory=list)
    top_k: int = 8
    include_sources: bool = True


class ExtractedField(BaseModel):
    name: str
    value: Any = None
    confidence: float = 0.0
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    reasoning: str | None = None


class ExtractResponse(BaseModel):
    document_id: str | None = None
    fields: list[ExtractedField]
    processing_ms: int = 0


class HighlightItem(BaseModel):
    chunk_id: str
    page: int | None = None
    text: str
    bbox: BBox | None = None
    score: float | None = None


class HighlightResponse(BaseModel):
    document_id: str
    highlights: list[HighlightItem]
    file_path: str | None = None


class EvalCase(BaseModel):
    query: str
    expected_contains: list[str] = Field(default_factory=list)
    expected_not_contains: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    document_id: str | None = None
    corpus_id: str | None = None


class EvalCaseResult(BaseModel):
    query: str
    passed: bool
    answer: str = ""
    top_chunk_ids: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    recall_at_k: float = 0.0
    confidence: float = 0.0


class EvalMetrics(BaseModel):
    total: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    avg_recall_at_k: float = 0.0
    avg_confidence: float = 0.0


class EvalRequest(BaseModel):
    mode: str = "agent"
    top_k: int = 5
    cases: list[EvalCase]


class EvalResponse(BaseModel):
    metrics: EvalMetrics
    results: list[EvalCaseResult]


class DocumentFamilyMember(BaseModel):
    document_id: str
    filename: str
    corpus_id: str | None = None
    document_family: str | None = None
    similarity: float = 0.0
    shared_signals: dict[str, Any] = Field(default_factory=dict)


class DocumentFamilyResponse(BaseModel):
    anchor_document_id: str | None = None
    family: str
    members: list[DocumentFamilyMember]
    signals: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    document_id_a: str
    document_id_b: str


class CompareResponse(BaseModel):
    document_id_a: str
    document_id_b: str
    shared_entities: list[dict[str, Any]] = Field(default_factory=list)
    only_in_a: list[dict[str, Any]] = Field(default_factory=list)
    only_in_b: list[dict[str, Any]] = Field(default_factory=list)
    table_diffs: list[dict[str, Any]] = Field(default_factory=list)
    field_diffs: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class ContradictionRequest(BaseModel):
    document_id_a: str | None = None
    document_id_b: str | None = None
    corpus_id: str | None = None


class ContradictionResponse(BaseModel):
    contradiction_count: int = 0
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    comparison_summary: dict[str, Any] | None = None


class GraphTraverseRequest(BaseModel):
    start_entity_query: str
    max_hops: int = 3
    corpus_id: str | None = None
    document_id: str | None = None
    predicate_filter: str | None = None


class GraphTraverseResponse(BaseModel):
    start_entities: list[Entity] = Field(default_factory=list)
    paths: list[list[dict[str, Any]]] = Field(default_factory=list)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class WebhookRegisterRequest(BaseModel):
    url: str
    events: list[str] = Field(default_factory=lambda: ["ingest.completed"])
    secret: str | None = None


class WebhookRegisterResponse(BaseModel):
    subscription_id: str
    url: str
    events: list[str]
    secret: str


class TemplateListResponse(BaseModel):
    templates: list[dict[str, Any]]


class TranslateRequest(BaseModel):
    text: str
    source_language: str = "auto"
    target_language: str = "en-IN"


class TranslateResponse(BaseModel):
    original: str
    translated: str
    source_language: str
    target_language: str

