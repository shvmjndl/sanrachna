from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.cost_tracker import CostTracker
from app.core.logging import get_logger
from app.db.init_db import SyncSessionLocal
from app.schemas.knowledge import ExtractRequest, ExtractResponse
from pipeline.extraction import SchemaExtractor
from pipeline.indexing import HybridIndexer
from services.llm import LLMService
from services.schema_templates import SchemaTemplateService

router = APIRouter()
settings = get_settings()
log = get_logger(__name__)
_templates = SchemaTemplateService()


@router.post("/extract", response_model=ExtractResponse)
async def extract_fields(req: ExtractRequest):
    if req.template_id and not req.fields:
        try:
            req = req.model_copy(update={"fields": _templates.get_fields(req.template_id)})
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown template: {req.template_id}")
    if not req.fields:
        raise HTTPException(status_code=400, detail="Provide fields or template_id")

    cost_tracker = CostTracker(SyncSessionLocal, log, buffer_writes=False)
    llm = LLMService(settings, cost_tracker)
    indexer = HybridIndexer(settings, llm, SyncSessionLocal, cost_tracker)
    extractor = SchemaExtractor(llm, indexer, SyncSessionLocal)
    return await extractor.extract(req)
