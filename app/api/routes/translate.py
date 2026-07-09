from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.schemas.knowledge import TranslateRequest, TranslateResponse
from services.translate import TranslateService

router = APIRouter()
settings = get_settings()


@router.post("/translate", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest):
    if not settings.sarvam_api_key:
        raise HTTPException(status_code=503, detail="Translation requires SARVAM_API_KEY")
    svc = TranslateService(settings.sarvam_api_key)
    result = svc.translate(req.text, req.source_language, req.target_language)
    return TranslateResponse(
        original=result.get("original", req.text),
        translated=result.get("translated", req.text),
        source_language=result.get("source_language", req.source_language),
        target_language=result.get("target_language", req.target_language),
    )
