from fastapi import APIRouter, HTTPException

from app.schemas.knowledge import TemplateListResponse
from services.schema_templates import SchemaTemplateService

router = APIRouter()
_templates = SchemaTemplateService()


@router.get("/templates", response_model=TemplateListResponse)
async def list_templates():
    return TemplateListResponse(templates=_templates.list_templates())


@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    try:
        fields = _templates.get_fields(template_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"template_id": template_id, "fields": [f.model_dump() for f in fields]}
