from fastapi import APIRouter, HTTPException

from app.db.init_db import SyncSessionLocal
from app.schemas.knowledge import ContradictionRequest, ContradictionResponse
from services.contradiction import ContradictionService

router = APIRouter()


@router.post("/contradictions", response_model=ContradictionResponse)
async def detect_contradictions(req: ContradictionRequest):
    service = ContradictionService(SyncSessionLocal)
    if req.document_id_a and req.document_id_b:
        result = service.detect_between(req.document_id_a, req.document_id_b)
        return ContradictionResponse(
            contradiction_count=result.get("contradiction_count", 0),
            contradictions=result.get("contradictions", []),
            comparison_summary=result.get("comparison_summary"),
        )
    if req.corpus_id:
        result = service.detect_in_corpus(req.corpus_id)
        return ContradictionResponse(
            contradiction_count=sum(c.get("contradiction_count", 0) for c in result.get("conflicts", [])),
            conflicts=result.get("conflicts", []),
        )
    raise HTTPException(status_code=400, detail="Provide document_id_a+b or corpus_id")
