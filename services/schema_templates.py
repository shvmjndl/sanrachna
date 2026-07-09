"""Named schema templates for schema-on-read extraction."""

from __future__ import annotations

from app.schemas.knowledge import FieldSpec

BUILTIN_TEMPLATES: dict[str, dict] = {
    "financial_summary": {
        "description": "Generic financial document — amounts, parties, dates",
        "fields": [
            FieldSpec(name="party_name", description="primary organization or person named as vendor, buyer, or issuer"),
            FieldSpec(name="total_amount", description="final payable or receivable amount including taxes"),
            FieldSpec(name="tax_amount", description="total tax or GST amount if present"),
            FieldSpec(name="document_date", description="primary date on the document"),
            FieldSpec(name="reference_id", description="invoice, order, or reference number"),
        ],
    },
    "identity_compliance": {
        "description": "IDs and regulatory numbers on any document",
        "fields": [
            FieldSpec(name="tax_id", description="GSTIN, VAT, or tax identification number"),
            FieldSpec(name="pan_or_tax_code", description="PAN or equivalent personal/business tax code"),
            FieldSpec(name="bank_code", description="IFSC, SWIFT, or bank routing code"),
            FieldSpec(name="registration_id", description="company or registration identifier"),
        ],
    },
    "tabular_charges": {
        "description": "Line-item charges from tables or stacked records",
        "fields": [
            FieldSpec(name="charge_name", description="name of the fee, charge, or line item asked about"),
            FieldSpec(name="base_amount", description="base amount before tax for the charge"),
            FieldSpec(name="tax_amount", description="tax amount for the charge"),
            FieldSpec(name="total_amount", description="total amount for the charge including tax"),
            FieldSpec(name="unit_or_quantity", description="quantity, units, or rate basis if present"),
        ],
    },
    "contract_terms": {
        "description": "Agreement terms and obligations",
        "fields": [
            FieldSpec(name="party_a", description="first party or client name"),
            FieldSpec(name="party_b", description="second party or vendor name"),
            FieldSpec(name="effective_date", description="start or effective date"),
            FieldSpec(name="payment_terms", description="payment terms or due period"),
            FieldSpec(name="termination_clause", description="termination or notice period if stated"),
        ],
    },
}


class SchemaTemplateService:
    def list_templates(self) -> list[dict]:
        return [
            {
                "template_id": tid,
                "description": meta["description"],
                "field_count": len(meta["fields"]),
                "fields": [f.model_dump() for f in meta["fields"]],
            }
            for tid, meta in BUILTIN_TEMPLATES.items()
        ]

    def get_fields(self, template_id: str) -> list[FieldSpec]:
        meta = BUILTIN_TEMPLATES.get(template_id)
        if not meta:
            raise KeyError(f"Unknown template: {template_id}")
        return meta["fields"]
