from fastapi import APIRouter
from sqlalchemy import func, select

from app.core.cost_tracker import CostLog
from app.db.init_db import SyncSessionLocal

router = APIRouter()


@router.get("/costs/summary")
async def cost_summary():
    with SyncSessionLocal() as session:
        sarvam = session.execute(
            select(func.coalesce(func.sum(CostLog.cost_inr), 0.0)).where(
                CostLog.service == "sarvam"
            )
        ).scalar()
        litellm_calls = session.execute(
            select(func.count()).where(CostLog.service == "litellm")
        ).scalar()
        total_tokens = session.execute(
            select(
                func.coalesce(func.sum(CostLog.input_tokens), 0),
                func.coalesce(func.sum(CostLog.output_tokens), 0),
            ).where(CostLog.service == "litellm")
        ).one()

        recent = session.execute(
            select(CostLog).order_by(CostLog.created_at.desc()).limit(20)
        ).scalars().all()

    return {
        "sarvam_cost_inr": float(sarvam or 0),
        "litellm_calls": litellm_calls or 0,
        "total_input_tokens": total_tokens[0] or 0,
        "total_output_tokens": total_tokens[1] or 0,
        "recent": [
            {
                "service": r.service,
                "operation": r.operation,
                "model": r.model,
                "cost_inr": r.cost_inr,
                "pages": r.pages,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in recent
        ],
    }
