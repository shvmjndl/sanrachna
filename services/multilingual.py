"""Multilingual query preprocessing — detect, translate, answer in user language."""

from __future__ import annotations

from app.core.config import Settings
from services.lid import LIDClient
from services.translate import TranslateService


class MultilingualQueryService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._lid = LIDClient(settings.sarvam_api_key) if settings.sarvam_api_key else None
        self._translate = TranslateService(settings.sarvam_api_key) if settings.sarvam_api_key else None

    def preprocess(self, query: str, target_search_language: str = "en-IN") -> dict:
        if not self._settings.auto_translate_queries or not self._translate:
            return {
                "original_query": query,
                "search_query": query,
                "query_language": "en-IN",
                "translated": False,
            }

        detected = "en-IN"
        if self._lid:
            try:
                detected, _ = self._lid.detect(query)
            except Exception:
                detected = "en-IN"

        if detected.split("-")[0] == target_search_language.split("-")[0]:
            return {
                "original_query": query,
                "search_query": query,
                "query_language": detected,
                "translated": False,
            }

        result = self._translate.translate(query, source_language="auto", target_language=target_search_language)
        return {
            "original_query": query,
            "search_query": result.get("translated", query),
            "query_language": result.get("source_language", detected),
            "answer_language": detected,
            "translated": True,
        }

    def postprocess_answer(self, answer: str, answer_language: str | None) -> str:
        if not answer_language or not self._translate or not self._settings.auto_translate_queries:
            return answer
        if answer_language.split("-")[0] == "en":
            return answer
        result = self._translate.translate(answer, source_language="en-IN", target_language=answer_language)
        return result.get("translated", answer)
