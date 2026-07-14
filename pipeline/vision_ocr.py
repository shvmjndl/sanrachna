"""Vision LLM OCR via LiteLLM — OpenAI, Gemini, or any configured VLM."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import fitz
import httpx

from app.core.config import Settings
from app.core.cost_tracker import CostTracker
from app.core.logging import get_logger
from pipeline.provenance import normalize_structured

log = get_logger(__name__)

DOCUMENT_OCR_PROMPT = (
    "Extract all information from the main body of this document image "
    "and represent it in markdown format, ignoring headers and footers. "
    "Tables should use HTML format, formulas LaTeX, in reading order. "
    "Output only the extracted content."
)


class VisionOCRDigitizer:
    """Page-by-page OCR using a vision-capable model through LiteLLM."""

    def __init__(self, settings: Settings, cost_tracker: CostTracker | None = None):
        self._settings = settings
        self._cost = cost_tracker
        self._base = settings.litellm_base_url.rstrip("/")
        self._api_key = settings.litellm_api_key
        self._models = self._resolve_models(settings)
        self._model = self._models[0]
        self._max_pages = settings.vision_ocr_max_pages
        self._scale = settings.vision_ocr_render_scale
        self._timeout = settings.vision_ocr_timeout_sec

    @staticmethod
    def _resolve_models(settings: Settings) -> list[str]:
        models: list[str] = []
        for name in [settings.vision_ocr_model, *settings.vision_ocr_fallback_models.split(",")]:
            name = name.strip()
            if name and name not in models:
                models.append(name)
        return models or ["gemini-3.5-flash"]

    def available(self) -> bool:
        if not self._settings.vision_ocr_enabled or not self._api_key:
            return False
        if any(m.startswith("gemini") for m in self._models):
            return bool(self._settings.gemini_api_key)
        return True

    def process_file(
        self,
        file_path: Path,
        language: str = "en",
        *,
        job_id: str | None = None,
        document_id: str | None = None,
    ) -> tuple[str, dict, int]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self._process_pdf(file_path, job_id=job_id, document_id=document_id)
        if ext in {".png", ".jpg", ".jpeg"}:
            text = self._ocr_image(file_path, job_id=job_id, document_id=document_id)
            md = f"## Page 1\n\n{text}".strip()
            structured = normalize_structured(
                {
                    "provider": "vision",
                    "model": self._model,
                    "pages": [{"page_number": 1, "blocks": [{"text": text}]}],
                },
                source="vision",
            )
            return md, structured, 1
        raise RuntimeError(f"Vision OCR does not support {ext}")

    def _process_pdf(
        self,
        pdf_path: Path,
        *,
        job_id: str | None = None,
        document_id: str | None = None,
    ) -> tuple[str, dict, int]:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        limit = min(page_count, self._max_pages)
        if page_count > limit:
            log.warning("vision_ocr_page_limit", total=page_count, limit=limit)

        pages_md: list[str] = []
        structured_pages: list[dict] = []

        with tempfile.TemporaryDirectory(prefix="sanrachna_vision_") as tmp:
            tmp_path = Path(tmp)
            for i in range(limit):
                page_num = i + 1
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(self._scale, self._scale))
                img_path = tmp_path / f"page_{page_num:04d}.png"
                pix.save(str(img_path))
                text = self._ocr_image(img_path, job_id=job_id, document_id=document_id)
                pages_md.append(f"## Page {page_num}\n\n{text}".strip())
                structured_pages.append(
                    {
                        "page_number": page_num,
                        "width": float(pix.width),
                        "height": float(pix.height),
                        "blocks": [{"text": text}],
                    }
                )
                log.info("vision_ocr_page_done", page=page_num, chars=len(text), model=self._model)

        doc.close()
        markdown = "\n\n---\n\n".join(pages_md)
        structured = normalize_structured(
            {"provider": "vision", "model": self._model, "pages": structured_pages},
            source="vision",
        )
        return markdown, structured, limit

    def _ocr_image(
        self,
        image_path: Path,
        *,
        job_id: str | None = None,
        document_id: str | None = None,
    ) -> str:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        data_url = f"data:{mime};base64,{encoded}"

        last_error: Exception | None = None
        for model in self._models:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                            {"type": "text", "text": DOCUMENT_OCR_PROMPT},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 4096,
            }
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(
                        f"{self._base}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                self._model = model
                usage = data.get("usage") or {}
                if self._cost:
                    self._cost.log_litellm(
                        job_id=job_id,
                        document_id=document_id,
                        model=model,
                        operation="vision_ocr",
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                    )
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                    content = "\n".join(parts)
                return str(content).strip()
            except Exception as exc:
                last_error = exc
                log.warning("vision_ocr_model_failed", model=model, error=str(exc))

        raise RuntimeError(f"Vision OCR failed for all models {self._models}: {last_error}")
