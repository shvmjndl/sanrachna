"""MinerU digitizer backend — subprocess CLI integration with graceful fallback."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger
from pipeline.provenance import normalize_structured

log = get_logger(__name__)


class MinerUDigitizer:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._bin = settings.mineru_bin

    def available(self) -> bool:
        return self._settings.mineru_enabled and shutil.which(self._bin) is not None

    def process_file(self, file_path: Path, language: str = "en") -> tuple[str, dict, int]:
        if not self.available():
            raise RuntimeError("MinerU CLI not available — install mineru or set MINERU_ENABLED=false")

        with tempfile.TemporaryDirectory(prefix="sanrachna_mineru_") as tmp:
            out_dir = Path(tmp) / "output"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                self._bin,
                "-p",
                str(file_path),
                "-o",
                str(out_dir),
                "-b",
                self._settings.mineru_backend,
            ]
            if language and language != "auto":
                cmd.extend(["-l", language.split("-")[0]])

            log.info("mineru_start", file=str(file_path), cmd=" ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self._settings.mineru_timeout_sec)
            if proc.returncode != 0:
                raise RuntimeError(f"MinerU failed: {proc.stderr[:500] or proc.stdout[:500]}")

            markdown, structured, pages = self._collect_output(out_dir, file_path.stem)
            structured = normalize_structured(structured, source="mineru")
            return markdown, structured, pages

    def _collect_output(self, out_dir: Path, stem: str) -> tuple[str, dict, int]:
        md_parts: list[str] = []
        pages_json: list[dict] = []

        for md_file in sorted(out_dir.rglob("*.md")):
            md_parts.append(md_file.read_text(encoding="utf-8", errors="replace"))

        for json_file in sorted(out_dir.rglob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                if "pages" in data:
                    pages_json.extend(data["pages"] if isinstance(data["pages"], list) else [data["pages"]])
                else:
                    pages_json.append(data)

        markdown = "\n\n---\n\n".join(md_parts).strip()
        if not markdown:
            markdown = f"[MinerU produced no markdown for {stem}]"

        structured = {"provider": "mineru", "pages": pages_json or [{"page_number": 1, "blocks": []}]}
        page_count = len(pages_json) or max(1, markdown.count("## Page"))
        return markdown, structured, page_count
