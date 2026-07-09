"""Sarvam Document Digitization adapter with 10-page batching."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import fitz
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.provenance import normalize_structured
from app.core.config import Settings

SARVAM_BASE = "https://api.sarvam.ai"


class SarvamClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._headers = {"api-subscription-key": settings.sarvam_api_key}

    def split_pdf(self, pdf_path: Path, max_pages: int) -> list[Path]:
        doc = fitz.open(pdf_path)
        if doc.page_count <= max_pages:
            doc.close()
            return [pdf_path]

        batches: list[Path] = []
        for start in range(0, doc.page_count, max_pages):
            end = min(start + max_pages, doc.page_count)
            batch = fitz.open()
            batch.insert_pdf(doc, from_page=start, to_page=end - 1)
            out = pdf_path.parent / f"{pdf_path.stem}_p{start + 1}-{end}.pdf"
            batch.save(out)
            batch.close()
            batches.append(out)
        doc.close()
        return batches

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def create_job(self, language: str, output_format: str = "md") -> str:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{SARVAM_BASE}/doc-digitization/job/v1",
                headers={**self._headers, "Content-Type": "application/json"},
                json={"job_parameters": {"language": language, "output_format": output_format}},
            )
            resp.raise_for_status()
            return resp.json()["job_id"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def upload_file(self, job_id: str, file_path: Path) -> None:
        with httpx.Client(timeout=120) as client:
            with open(file_path, "rb") as f:
                resp = client.post(
                    f"{SARVAM_BASE}/doc-digitization/job/v1/upload-files",
                    headers=self._headers,
                    data={"job_id": job_id},
                    files={"file": (file_path.name, f)},
                )
            resp.raise_for_status()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def start_job(self, job_id: str) -> None:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{SARVAM_BASE}/doc-digitization/job/v1/{job_id}/start",
                headers=self._headers,
            )
            resp.raise_for_status()

    def wait_for_completion(self, job_id: str, poll_interval: float = 3.0, timeout: float = 600) -> dict:
        import time

        elapsed = 0.0
        with httpx.Client(timeout=30) as client:
            while elapsed < timeout:
                resp = client.get(
                    f"{SARVAM_BASE}/doc-digitization/job/v1/{job_id}/status",
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()
                state = data.get("job_state", "")
                if state in ("Completed", "PartiallyCompleted"):
                    return data
                if state == "Failed":
                    raise RuntimeError(f"Sarvam job {job_id} failed: {data}")
                time.sleep(poll_interval)
                elapsed += poll_interval
        raise TimeoutError(f"Sarvam job {job_id} timed out after {timeout}s")

    def download_output(self, job_id: str) -> bytes:
        with httpx.Client(timeout=120) as client:
            links_resp = client.get(
                f"{SARVAM_BASE}/doc-digitization/job/v1/{job_id}/download",
                headers=self._headers,
            )
            links_resp.raise_for_status()
            links = links_resp.json()

            download_url = links.get("download_url") or links.get("output_url")
            if not download_url and "files" in links:
                download_url = links["files"][0].get("url")

            if download_url:
                file_resp = client.get(download_url)
                file_resp.raise_for_status()
                return file_resp.content

            # Fallback: direct download endpoint
            resp = client.get(
                f"{SARVAM_BASE}/doc-digitization/job/v1/{job_id}/download-output",
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.content

    def process_file(self, file_path: Path, language: str) -> tuple[str, dict, list[str], int]:
        """Returns (markdown, structured_json, job_ids, pages_processed)."""
        batches = (
            self.split_pdf(file_path, self._settings.sarvam_max_pages)
            if file_path.suffix.lower() == ".pdf"
            else [file_path]
        )

        all_markdown: list[str] = []
        all_json: list[dict] = []
        job_ids: list[str] = []
        total_pages = 0

        for batch in batches:
            job_id = self.create_job(language)
            self.upload_file(job_id, batch)
            self.start_job(job_id)
            status = self.wait_for_completion(job_id)
            job_ids.append(job_id)

            metrics = status.get("page_metrics", {})
            total_pages += metrics.get("pages_processed", metrics.get("pages_succeeded", 0))

            output_bytes = self.download_output(job_id)
            md, structured = self._parse_output_zip(output_bytes)
            all_markdown.append(md)
            all_json.append(structured)

        combined_md = "\n\n---\n\n".join(all_markdown)
        combined_json = normalize_structured({"pages": all_json}, source="sarvam")
        return combined_md, combined_json, job_ids, total_pages

    def _parse_output_zip(self, data: bytes) -> tuple[str, dict]:
        md_parts: list[str] = []
        structured: dict = {}

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                content = zf.read(name).decode("utf-8", errors="replace")
                if name.endswith(".md"):
                    md_parts.append(content)
                elif name.endswith(".json"):
                    try:
                        structured = json.loads(content)
                    except json.JSONDecodeError:
                        structured = {"raw": content}

        return "\n\n".join(md_parts), structured
