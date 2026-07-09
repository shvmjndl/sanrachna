"""India-specific compliance validation for documents."""

from __future__ import annotations

import re

GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]$")
GSTIN_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PAN_RE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
AADHAAR_RE = re.compile(r"^\d{12}$")


def _gstin_checksum_valid(gstin: str) -> bool:
    """Validate GSTIN check digit (15th character) per GST portal algorithm."""
    if len(gstin) != 15:
        return False
    factor = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    total = 0
    for i in range(14):
        try:
            code_point = GSTIN_CHARSET.index(gstin[i])
        except ValueError:
            return False
        digit = factor[i] * code_point
        digit = (digit // 36) + (digit % 36)
        total += digit
    check = (36 - (total % 36)) % 36
    return GSTIN_CHARSET[check] == gstin[14]


def validate_gstin(gstin: str) -> dict:
    clean = re.sub(r"\s+", "", gstin).upper()
    valid_format = bool(GSTIN_RE.match(clean))
    checksum_ok = _gstin_checksum_valid(clean) if valid_format else False
    valid = valid_format and checksum_ok
    issue = None
    if not valid_format:
        issue = "Invalid GSTIN format (expected 15 chars: 2 digits + 5 letters + 4 digits + 1 letter + 1 digit + Z + 1 alphanumeric)"
    elif not checksum_ok:
        issue = "Invalid GSTIN checksum digit"
    return {
        "value": clean,
        "type": "GSTIN",
        "valid": valid,
        "issue": issue,
    }


def validate_pan(pan: str) -> dict:
    clean = re.sub(r"\s+", "", pan).upper()
    valid_format = bool(PAN_RE.match(clean))
    return {
        "value": clean,
        "type": "PAN",
        "valid": valid_format,
        "issue": None if valid_format else "Invalid PAN format (expected AAAAA9999A)",
    }


def validate_ifsc(ifsc: str) -> dict:
    clean = re.sub(r"\s+", "", ifsc).upper()
    valid_format = bool(IFSC_RE.match(clean))
    return {
        "value": clean,
        "type": "IFSC",
        "valid": valid_format,
        "issue": None if valid_format else "Invalid IFSC format (expected 4 letters + 0 + 6 alphanumeric)",
    }


def validate_aadhaar(aadhaar: str) -> dict:
    clean = re.sub(r"\s+", "", aadhaar)
    valid_format = bool(AADHAAR_RE.match(clean))
    return {
        "value": clean,
        "type": "AADHAAR",
        "valid": valid_format,
        "issue": None if valid_format else "Invalid Aadhaar format (expected 12 digits)",
    }


VALIDATORS = {
    "GSTIN": validate_gstin,
    "PAN": validate_pan,
    "IFSC": validate_ifsc,
    "AADHAAR": validate_aadhaar,
}


def check_entity(entity_type: str, value: str) -> dict:
    fn = VALIDATORS.get(entity_type.upper())
    if not fn:
        return {"value": value, "type": entity_type, "valid": None, "issue": "No validator for this type"}
    return fn(value)


def check_document_compliance(entities: list[dict]) -> dict:
    """Run compliance checks on all regulatory entities in a document."""
    results = []
    summary = {"total": 0, "valid": 0, "invalid": 0, "unchecked": 0}

    for ent in entities:
        etype = ent.get("type", "")
        if etype not in VALIDATORS:
            continue
        summary["total"] += 1
        check = check_entity(etype, ent.get("text", ""))
        results.append({**check, "entity_id": ent.get("id"), "source_chunk": ent.get("source_chunk")})
        if check["valid"] is True:
            summary["valid"] += 1
        elif check["valid"] is False:
            summary["invalid"] += 1
        else:
            summary["unchecked"] += 1

    return {
        "summary": summary,
        "compliant": summary["invalid"] == 0 and summary["total"] > 0,
        "checks": results,
    }
