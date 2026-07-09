from fastapi import APIRouter

from app.core.config import get_settings
from app.core.cost_tracker import CostTracker
from app.core.logging import get_logger
from app.db.init_db import SyncSessionLocal
from app.schemas.knowledge import EvalRequest, EvalResponse
from eval.harness import EvalHarness

router = APIRouter()
settings = get_settings()
log = get_logger(__name__)


@router.post("/eval", response_model=EvalResponse)
async def run_eval(req: EvalRequest):
    harness = EvalHarness(settings, SyncSessionLocal, log)
    return await harness.run(req)
