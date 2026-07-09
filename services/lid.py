"""Sarvam Language Identification client."""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

SARVAM_LID_URL = "https://api.sarvam.ai/text-lid"


class LIDClient:
    def __init__(self, api_key: str):
        self._headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def detect(self, text: str) -> tuple[str, str]:
        sample = text[:1000].strip()
        if not sample:
            return "hi-IN", "Deva"

        with httpx.Client(timeout=15) as client:
            resp = client.post(
                SARVAM_LID_URL,
                headers=self._headers,
                json={"input": sample},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("language_code", "hi-IN"), data.get("script_code", "Deva")

    def detect_pages(self, pages: list[str], fallback: str = "hi-IN") -> list[str]:
        """Detect language per page, return list of language codes."""
        languages: list[str] = []
        for page_text in pages:
            if len(page_text.strip()) < 20:
                languages.append(fallback)
                continue
            try:
                lang, _ = self.detect(page_text)
                languages.append(lang)
            except Exception:
                languages.append(fallback)
        return languages
