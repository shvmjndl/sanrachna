"""File type routing for mixed corpora."""

from __future__ import annotations

import mimetypes
import zipfile
from pathlib import Path

SARVAM_TYPES = {".pdf", ".png", ".jpg", ".jpeg"}
CONVERTIBLE_TYPES = {".txt", ".md", ".csv", ".eml", ".docx", ".pptx"}
MEDIA_TYPES = {".mp4", ".mov", ".mp3", ".wav", ".m4a"}


def detect_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def classify_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SARVAM_TYPES:
        return "sarvam"
    if ext in CONVERTIBLE_TYPES:
        return "convert"
    if ext in MEDIA_TYPES:
        return "media"
    if ext == ".zip":
        return "archive"
    return "unsupported"


def extract_archive(archive_path: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX"):
                continue
            target = dest / Path(info.filename).name
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            files.append(target)
    return files


def collect_files(root: Path) -> list[Path]:
    if root.suffix.lower() == ".zip":
        extracted = extract_archive(root, root.parent / f"{root.stem}_extracted")
        return [f for f in extracted if f.is_file()]
    return [root]
