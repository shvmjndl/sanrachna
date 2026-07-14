"""Content-aware digitizer routing — pick best available backend."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

_INDIC_RE = re.compile(r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]")


@dataclass
class DigitizerChoice:
    provider: str
    reason: str
    text_density: float = 0.0
    indic_ratio: float = 0.0
    page_count: int = 0


def _pdf_probe(file_path: Path) -> tuple[float, float, int, str]:
    try:
        import fitz
    except ImportError:
        return 0.0, 0.0, 0, ""

    doc = fitz.open(file_path)
    pages = doc.page_count
    texts: list[str] = []
    for i in range(min(pages, 5)):
        texts.append(doc[i].get_text("text"))
    doc.close()

    combined = "\n".join(texts)
    chars = len(combined.strip())
    density = chars / max(1, min(pages, 5) * 500)
    indic = len(_INDIC_RE.findall(combined))
    indic_ratio = indic / max(1, len(combined))
    return density, indic_ratio, pages, combined[:2000]


def _first_ocr_provider(
    *,
    mineru_available: bool,
    sarvam_available: bool,
    vision_available: bool,
    tesseract_available: bool = False,
) -> str | None:
    if mineru_available:
        return "mineru"
    if sarvam_available:
        return "sarvam"
    if vision_available:
        return "vision"
    if tesseract_available:
        return "tesseract"
    return None


def choose_digitizer(
    file_path: Path,
    *,
    provider_setting: str = "auto",
    sarvam_available: bool = False,
    mineru_available: bool = False,
    vision_available: bool = False,
    tesseract_available: bool = False,
    language: str = "en-IN",
) -> DigitizerChoice:
    ext = file_path.suffix.lower()
    setting = provider_setting.lower()

    if setting == "vision":
        return DigitizerChoice(
            "vision" if vision_available else "local",
            "digitization_provider=vision" if vision_available else "vision unavailable",
        )
    if setting == "sarvam":
        return DigitizerChoice("sarvam", "digitization_provider=sarvam")
    if setting == "mineru":
        return DigitizerChoice(
            "mineru" if mineru_available else "local",
            "digitization_provider=mineru" if mineru_available else "mineru unavailable",
        )
    if setting == "local":
        # Still probe — orchestrator will fallback to OCR if local yields nothing
        if ext == ".pdf":
            density, indic_ratio, pages, _ = _pdf_probe(file_path)
            if density < 0.08:
                ocr = _first_ocr_provider(
                    mineru_available=mineru_available,
                    sarvam_available=sarvam_available,
                    vision_available=vision_available,
                    tesseract_available=tesseract_available,
                )
                if ocr:
                    return DigitizerChoice(
                        ocr,
                        f"scanned PDF — prefer {ocr} despite digitization_provider=local",
                        density,
                        indic_ratio,
                        pages,
                    )
        return DigitizerChoice("local", "digitization_provider=local")

    if ext in {".png", ".jpg", ".jpeg"}:
        ocr = _first_ocr_provider(
            mineru_available=False,
            sarvam_available=sarvam_available,
            vision_available=vision_available,
            tesseract_available=tesseract_available,
        )
        if ocr:
            return DigitizerChoice(ocr, f"image OCR via {ocr}")
        return DigitizerChoice("local", "image without OCR backend")

    if ext != ".pdf":
        return DigitizerChoice("local", "non-pdf convertible format")

    density, indic_ratio, pages, _ = _pdf_probe(file_path)

    if density < 0.08:
        ocr = _first_ocr_provider(
            mineru_available=mineru_available,
            sarvam_available=sarvam_available,
            vision_available=vision_available,
            tesseract_available=tesseract_available,
        )
        if ocr:
            return DigitizerChoice(
                ocr,
                f"low text density — {ocr}",
                density,
                indic_ratio,
                pages,
            )

    if indic_ratio > 0.05 and sarvam_available:
        return DigitizerChoice("sarvam", "Indic script detected", density, indic_ratio, pages)
    if indic_ratio > 0.05 and vision_available:
        return DigitizerChoice("vision", "Indic script — vision OCR", density, indic_ratio, pages)

    if pages > 15 and mineru_available and density < 0.35:
        return DigitizerChoice("mineru", "long complex PDF", density, indic_ratio, pages)
    if mineru_available and density < 0.2:
        return DigitizerChoice("mineru", "moderate density complex PDF", density, indic_ratio, pages)
    if sarvam_available and language.endswith("-IN") and language != "en-IN":
        return DigitizerChoice("sarvam", "non-english Indic locale hint", density, indic_ratio, pages)

    return DigitizerChoice("local", "text-based PDF", density, indic_ratio, pages)


def mineru_cli_available(mineru_bin: str = "mineru") -> bool:
    return shutil.which(mineru_bin) is not None
