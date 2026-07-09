from fastapi import APIRouter

from app.db.init_db import SyncSessionLocal
from app.schemas.knowledge import CompareRequest, CompareResponse
from services.compare import CompareService

router = APIRouter()


@router.post("/compare", response_model=CompareResponse)
async def compare_documents(req: CompareRequest):
    result = CompareService(SyncSessionLocal).compare(req.document_id_a, req.document_id_b)
    if result.get("error"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=result["error"])
    return CompareResponse(**result)
