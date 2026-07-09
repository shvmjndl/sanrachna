"""Sarvam translation for cross-lingual agent queries."""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"


class TranslateService:
    def __init__(self, api_key: str):
        self._headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "en-IN",
    ) -> dict:
        with httpx.Client(timeout=20) as client:
            resp = client.post(
                SARVAM_TRANSLATE_URL,
                headers=self._headers,
                json={
                    "input": text[:2000],
                    "source_language_code": source_language,
                    "target_language_code": target_language,
                    "mode": "classic",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        translated = data.get("translated_text") or data.get("output") or text
        return {
            "original": text,
            "translated": translated,
            "source_language": source_language,
            "target_language": target_language,
        }
