"""Vision LLM OCR via LiteLLM — OpenAI, Gemini, or any configured VLM."""

from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fitz
import httpx

from app.core.config import Settings
from app.core.cost_tracker import CostTracker
from app.core.logging import get_logger
from pipeline.page_parallel import map_parallel
from pipeline.provenance import normalize_structured

log = get_logger(__name__)

DOCUMENT_OCR_PROMPT = (
    "Extract all information from the main body of this document image "
    "and represent it in markdown format, ignoring headers and footers. "
    "Tables should use HTML format, formulas LaTeX, in reading order. "
    "Output only the extracted content."
)


@dataclass
class _PageRender:
    page_num: int
    img_path: Path
    width: float
    height: float


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
        self._concurrency = settings.vision_ocr_concurrency
        self._image_detail = settings.vision_ocr_image_detail
        self._client = httpx.Client(timeout=self._timeout)

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

    def close(self) -> None:
        self._client.close()

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

        with tempfile.TemporaryDirectory(prefix="sanrachna_vision_") as tmp:
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
                text = self._ocr_image(render.img_path, job_id=job_id, document_id=document_id)
                log.info("vision_ocr_page_done", page=render.page_num, chars=len(text), model=self._model)
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
                            {"type": "image_url", "image_url": {"url": data_url, "detail": self._image_detail}},
                            {"type": "text", "text": DOCUMENT_OCR_PROMPT},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 4096,
            }
            try:
                resp = self._client.post(
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
