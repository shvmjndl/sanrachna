"""Shared digitizer helpers — output quality checks and fallback ordering."""

from __future__ import annotations

_PLACEHOLDER_MARKERS = (
    "no extractable text",
    "cannot ocr images",
    "mineru produced no",
    "unsupported in local mode",
    "set sarvam_api_key",
)


def is_usable_markdown(markdown: str, *, min_chars: int = 80) -> bool:
    """Return False for empty output or local placeholder messages."""
    text = (markdown or "").strip()
    if len(text) < min_chars:
        return False
    lower = text.lower()
    return not any(marker in lower for marker in _PLACEHOLDER_MARKERS)


def build_fallback_chain(
    primary: str,
    *,
    mineru_available: bool = False,
    sarvam_available: bool = False,
    vision_available: bool = False,
    tesseract_available: bool = False,
    include_local: bool = True,
) -> list[str]:
    """Order digitizers: primary first, then OCR backends, local last."""
    chain: list[str] = []

    def add(provider: str | None) -> None:
        if provider and provider not in chain:
            chain.append(provider)

    add(primary)
    for provider in ("mineru", "sarvam", "vision", "tesseract"):
        if provider == "mineru" and mineru_available:
            add("mineru")
        elif provider == "sarvam" and sarvam_available:
            add("sarvam")
        elif provider == "vision" and vision_available:
            add("vision")
        elif provider == "tesseract" and tesseract_available:
            add("tesseract")
    if include_local:
        add("local")
    return chain
