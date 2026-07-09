from fastapi import APIRouter

from app.db.init_db import SyncSessionLocal
from app.schemas.knowledge import WebhookRegisterRequest, WebhookRegisterResponse
from services.webhooks import WebhookService

router = APIRouter()


@router.post("/webhooks", response_model=WebhookRegisterResponse)
async def register_webhook(req: WebhookRegisterRequest):
    result = WebhookService(SyncSessionLocal).register(req.url, req.events, req.secret)
    return WebhookRegisterResponse(**result)


@router.get("/webhooks")
async def list_webhooks():
    return WebhookService(SyncSessionLocal).list_subscriptions()
