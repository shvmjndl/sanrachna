"""Local Tesseract OCR for scanned PDFs when cloud vision is unavailable."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fitz

from app.core.config import Settings
from app.core.logging import get_logger
from pipeline.page_parallel import map_parallel
from pipeline.provenance import normalize_structured

log = get_logger(__name__)


@dataclass
class _PageRender:
    page_num: int
    img_path: Path
    width: float
    height: float


class TesseractOCRDigitizer:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._max_pages = settings.tesseract_ocr_max_pages
        self._scale = settings.tesseract_ocr_render_scale
        self._lang = settings.tesseract_ocr_lang
        self._concurrency = settings.tesseract_ocr_concurrency

    def available(self) -> bool:
        if not self._settings.tesseract_ocr_enabled:
            return False
        if not shutil.which("tesseract"):
            return False
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return False
        return True

    def process_file(self, file_path: Path, language: str = "en") -> tuple[str, dict, int]:
        import pytesseract

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self._process_pdf(file_path, pytesseract)
        if ext in {".png", ".jpg", ".jpeg"}:
            text = pytesseract.image_to_string(str(file_path), lang=self._lang).strip()
            md = f"## Page 1\n\n{text}".strip()
            structured = normalize_structured(
                {
                    "provider": "tesseract",
                    "pages": [{"page_number": 1, "blocks": [{"text": text}]}],
                },
                source="tesseract",
            )
            return md, structured, 1
        raise RuntimeError(f"Tesseract OCR does not support {ext}")

    def _process_pdf(self, pdf_path: Path, pytesseract) -> tuple[str, dict, int]:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        limit = min(page_count, self._max_pages)
        if page_count > limit:
            log.warning("tesseract_page_limit", total=page_count, limit=limit)

        with tempfile.TemporaryDirectory(prefix="sanrachna_tesseract_") as tmp:
            tmp_path = Path(tmp)
            renders: list[_PageRender] = []
            for i in range(limit):
                page_num = i + 1
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(self._scale, self._scale))
                img_path = tmp_path / f"page_{page_num:04d}.png"
                pix.save(str(img_path))
                renders.append(
                    _PageRender(page_num=page_num, img_path=img_path, width=float(pix.width), height=float(pix.height))
                )
            doc.close()

            def ocr_page(render: _PageRender) -> tuple[int, str]:
                text = pytesseract.image_to_string(str(render.img_path), lang=self._lang).strip()
                log.info("tesseract_page_done", page=render.page_num, chars=len(text))
                return render.page_num, text

            ocr_by_page = dict(map_parallel(renders, ocr_page, max_workers=self._concurrency))

        pages_md: list[str] = []
        structured_pages: list[dict] = []
        for render in renders:
            text = ocr_by_page[render.page_num]
            pages_md.append(f"## Page {render.page_num}\n\n{text}".strip())
            structured_pages.append(
                {
                    "page_number": render.page_num,
                    "width": render.width,
                    "height": render.height,
                    "blocks": [{"text": text}],
                }
            )

        markdown = "\n\n---\n\n".join(pages_md)
        structured = normalize_structured(
            {"provider": "tesseract", "pages": structured_pages},
            source="tesseract",
        )
        return markdown, structured, limit
