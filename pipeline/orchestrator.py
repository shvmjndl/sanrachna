"""End-to-end document processing orchestrator."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from app.core.config import Settings
from app.core.cost_tracker import CostTracker
from app.core.logging import get_logger
from app.schemas.knowledge import (
    Chunk,
    ChunkType,
    EmbeddingsInfo,
    KnowledgeGraph,
    KnowledgeObject,
    ProcessingMetadata,
    SourceInfo,
)
from pipeline.chunking import chunk_markdown, get_indexable_chunks
from pipeline.classifier import classify_knowledge_object
from pipeline.converters import convert_to_markdown
from pipeline.digitizer_router import choose_digitizer, mineru_cli_available
from pipeline.enrichment import caption_images, extract_image_refs, extract_tables, tables_to_search_chunks
from pipeline.entities import extract_entities
from pipeline.graph import extract_relationships
from pipeline.indexing import HybridIndexer
from pipeline.local_digitizer import LocalDigitizer
from pipeline.mineru import MinerUDigitizer
from pipeline.quality import enrich_knowledge_object
from pipeline.router import classify_file, detect_mime
from pipeline.sarvam import SarvamClient
from pipeline.table_merge import merge_cross_page_tables
from services.entity_resolution import EntityResolver
from services.lid import LIDClient
from services.llm import LLMService

log = get_logger(__name__)


class PipelineOrchestrator:
    def __init__(
        self,
        settings: Settings,
        cost_tracker: CostTracker,
        session_factory=None,
    ):
        self._settings = settings
        self._cost = cost_tracker
        self._sarvam = SarvamClient(settings) if settings.sarvam_api_key else None
        self._local = LocalDigitizer()
        self._mineru = MinerUDigitizer(settings)
        self._llm = LLMService(settings, cost_tracker)
        self._indexer = HybridIndexer(settings, self._llm, session_factory, cost_tracker)
        self._entity_resolver = EntityResolver(session_factory) if session_factory else None
        self._lid = LIDClient(settings.sarvam_api_key) if settings.sarvam_api_key else None

    async def process_file(
        self,
        file_path: Path,
        language: str,
        job_id: str,
        corpus_id: str | None = None,
        options: dict | None = None,
        document_id: str | None = None,
        supersedes_id: str | None = None,
    ) -> KnowledgeObject:
        options = options or {}
        start = time.monotonic()
        document_id = document_id or f"doc_{uuid.uuid4().hex[:12]}"
        file_type = classify_file(file_path)

        choice = choose_digitizer(
            file_path,
            provider_setting=self._settings.digitization_provider,
            sarvam_available=self._sarvam is not None,
            mineru_available=self._mineru.available() or mineru_cli_available(self._settings.mineru_bin),
            language=language,
        )

        log.info(
            "processing_file",
            document_id=document_id,
            file=str(file_path),
            type=file_type,
            digitizer=choice.provider,
            reason=choice.reason,
        )

        markdown = ""
        structured: dict = {}
        sarvam_job_ids: list[str] = []
        pages = 0
        detected_languages: list[str] = []
        digitizer_used = choice.provider

        if file_type == "sarvam":
            markdown, structured, sarvam_job_ids, pages, detected_languages = self._digitize_pdf_image(
                file_path, language, choice, options, job_id, document_id
            )
        elif file_type == "convert":
            digitizer_used = "local"
            markdown = convert_to_markdown(file_path)
            pages = 1
            if options.get("auto_detect_language", True) and self._lid and markdown:
                try:
                    language, _ = self._lid.detect(markdown)
                    detected_languages.append(language)
                except Exception:
                    pass
            elif not detected_languages:
                detected_languages.append(language or "en-IN")
        elif file_type == "media":
            markdown = f"[Media file: {file_path.name} — STT processing pending]"
        else:
            markdown = f"[Unsupported file type: {file_path.suffix}]"

        all_chunks = chunk_markdown(markdown, self._settings, structured)
        tables = merge_cross_page_tables(extract_tables(markdown))
        table_texts = tables_to_search_chunks(tables)
        if table_texts:
            parent_id = next(
                (c.id for c in all_chunks if c.chunk_type == ChunkType.PARENT),
                f"parent_{uuid.uuid4().hex[:8]}",
            )
            parent_text = next((c.text for c in all_chunks if c.id == parent_id), markdown)
            for text in table_texts:
                all_chunks.append(
                    Chunk(
                        id=f"child_{uuid.uuid4().hex[:8]}",
                        text=text,
                        section="record",
                        token_count=max(1, len(text) // 4),
                        parent_id=parent_id,
                        parent_text=parent_text,
                        chunk_type=ChunkType.CHILD,
                    )
                )
        indexable = get_indexable_chunks(all_chunks)
        images = extract_image_refs(markdown, structured)

        entities = []
        if options.get("enable_entities", True):
            entities = await extract_entities(
                self._llm, indexable, language, job_id=job_id, document_id=document_id
            )
            if self._entity_resolver:
                entities = self._entity_resolver.resolve(entities, document_id, corpus_id)

        relationships = []
        graph = None
        if options.get("enable_graph", True) and entities:
            relationships, graph = await extract_relationships(
                self._llm, entities, markdown, job_id=job_id, document_id=document_id
            )

        if options.get("enable_captions", True) and images:
            images = await caption_images(
                self._llm, images, markdown[:1000], job_id=job_id, document_id=document_id
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        ko = KnowledgeObject(
            document_id=document_id,
            source=SourceInfo(
                filename=file_path.name,
                path=str(file_path),
                mime_type=detect_mime(file_path),
                language=language,
                detected_type=file_type,
                detected_languages=detected_languages,
                family_signals={"digitizer": digitizer_used, "digitizer_reason": choice.reason},
            ),
            markdown=markdown,
            chunks=all_chunks,
            tables=tables,
            images=images,
            entities=entities,
            relationships=relationships,
            knowledge_graph=graph or KnowledgeGraph(),
            embeddings=None,
            metadata=ProcessingMetadata(
                pages=pages,
                processing_time_ms=elapsed_ms,
                sarvam_job_ids=sarvam_job_ids,
                language=language,
                detected_languages=detected_languages,
            ),
        )
        ko = classify_knowledge_object(ko)
        ko = enrich_knowledge_object(ko)

        indexable = get_indexable_chunks(ko.chunks)
        embeddings_info = None
        if indexable:
            vectors = await self._indexer.embed_chunks(indexable, job_id, document_id)
            indexed = await self._indexer.index_chunks(
                document_id, indexable, vectors, corpus_id=corpus_id, job_id=job_id
            )
            embeddings_info = EmbeddingsInfo(
                model=self._settings.litellm_embedding_model,
                dimensions=len(vectors[0]) if vectors else 0,
                chunk_count=indexed,
            )
        ko.embeddings = embeddings_info
        return ko

    def _digitize_pdf_image(
        self,
        file_path: Path,
        language: str,
        choice,
        options: dict,
        job_id: str,
        document_id: str,
    ) -> tuple[str, dict, list[str], int, list[str]]:
        detected_languages: list[str] = []
        sarvam_job_ids: list[str] = []
        provider = choice.provider

        if options.get("auto_detect_language", True) and self._lid:
            sample_text = self._sample_file_text(file_path)
            if sample_text:
                try:
                    language, _ = self._lid.detect(sample_text)
                    detected_languages.append(language)
                except Exception:
                    pass

        chain = [provider, "sarvam" if self._sarvam else None, "local"]
        tried: set[str] = set()
        last_error = None

        for p in chain:
            if not p or p in tried:
                continue
            tried.add(p)
            try:
                if p == "mineru" and self._mineru.available():
                    md, structured, pages = self._mineru.process_file(file_path, language)
                    return md, structured, sarvam_job_ids, pages, detected_languages
                if p == "sarvam" and self._sarvam:
                    md, structured, sarvam_job_ids, pages = self._sarvam.process_file(file_path, language)
                    self._cost.log_sarvam(
                        job_id=job_id,
                        document_id=document_id,
                        pages=pages,
                        cost_per_page=self._settings.sarvam_cost_per_page_inr,
                        extra={"sarvam_job_ids": sarvam_job_ids},
                    )
                    return md, structured, sarvam_job_ids, pages, detected_languages
                if p == "local":
                    md, structured, pages = self._local.process_file(file_path)
                    if not detected_languages:
                        detected_languages.append(language or "en-IN")
                    return md, structured, sarvam_job_ids, pages, detected_languages
            except Exception as exc:
                last_error = exc
                log.warning("digitizer_fallback", provider=p, error=str(exc))

        raise RuntimeError(f"All digitizers failed: {last_error}")

    def _sample_file_text(self, file_path: Path) -> str:
        if file_path.suffix.lower() in (".txt", ".md", ".csv", ".eml"):
            return file_path.read_text(encoding="utf-8", errors="replace")[:1000]
        if file_path.suffix.lower() == ".pdf":
            try:
                import fitz

                doc = fitz.open(file_path)
                text = doc[0].get_text() if doc.page_count > 0 else ""
                doc.close()
                return text[:1000]
            except Exception:
                return ""
        return ""
