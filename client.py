"""Sanrachna Python SDK — typed HTTP client for RAG, agents, extraction, and eval."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterator

import requests

DEFAULT_API_URL = os.environ.get("SANRACHNA_API_URL", "http://localhost:8000")


class SanrachnaError(RuntimeError):
    pass


class SanrachnaClient:
    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            resp = self.session.request(method, self._url(path), timeout=self.timeout, **kwargs)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except requests.RequestException as exc:
            raise SanrachnaError(str(exc)) from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def ingest(
        self,
        file_path: Path,
        *,
        language: str = "en-IN",
        enable_entities: bool = True,
        enable_graph: bool = True,
        enable_captions: bool = True,
        auto_detect_language: bool = False,
        document_id: str | None = None,
        supersedes_id: str | None = None,
        incremental: bool = True,
    ) -> dict[str, Any]:
        data = {
            "language": language,
            "enable_entities": str(enable_entities).lower(),
            "enable_graph": str(enable_graph).lower(),
            "enable_captions": str(enable_captions).lower(),
            "auto_detect_language": str(auto_detect_language).lower(),
            "incremental": str(incremental).lower(),
        }
        if document_id:
            data["document_id"] = document_id
        if supersedes_id:
            data["supersedes_id"] = supersedes_id
        with file_path.open("rb") as f:
            return self._request(
                "POST",
                "/v1/ingest",
                files={"file": (file_path.name, f)},
                data=data,
            )

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/jobs/{job_id}")

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        on_progress=None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_job(job_id)
            if on_progress:
                on_progress(status)
            state = status.get("status")
            if state == "completed":
                return status
            if state == "failed":
                raise SanrachnaError(status.get("error") or "Ingest job failed")
            time.sleep(poll_interval)
        raise SanrachnaError(f"Job {job_id} timed out after {timeout}s")

    def list_documents(self, *, corpus_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if corpus_id:
            params["corpus_id"] = corpus_id
        return self._request("GET", "/v1/documents", params=params)

    def get_document(self, document_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/documents/{document_id}")

    def delete_document(self, document_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/documents/{document_id}")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
        corpus_id: str | None = None,
        hybrid: bool = True,
        rerank: bool = True,
        expand_parent: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "hybrid": hybrid,
            "rerank": rerank,
            "expand_parent": expand_parent,
        }
        if document_id:
            payload["document_id"] = document_id
        if corpus_id:
            payload["corpus_id"] = corpus_id
        return self._request("POST", "/v1/search", json=payload)

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        max_steps: int = 8,
        document_id: str | None = None,
        corpus_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": question,
            "top_k": top_k,
            "max_steps": max_steps,
        }
        if document_id:
            payload["document_id"] = document_id
        if corpus_id:
            payload["corpus_id"] = corpus_id
        return self._request("POST", "/v1/query", json=payload)

    def extract(
        self,
        fields: list[dict[str, str]],
        *,
        document_id: str | None = None,
        corpus_id: str | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/extract",
            json={
                "document_id": document_id,
                "corpus_id": corpus_id,
                "top_k": top_k,
                "fields": fields,
            },
        )

    def highlights(self, document_id: str, chunk_ids: list[str]) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/v1/documents/{document_id}/highlights",
            params={"chunk_ids": ",".join(chunk_ids)},
        )

    def families(
        self,
        *,
        document_id: str | None = None,
        corpus_id: str | None = None,
        min_similarity: float = 0.55,
        limit: int = 20,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"min_similarity": min_similarity, "limit": limit}
        if document_id:
            params["document_id"] = document_id
        if corpus_id:
            params["corpus_id"] = corpus_id
        return self._request("GET", "/v1/families", params=params)

    def eval_cases(
        self,
        cases: list[dict[str, Any]],
        *,
        mode: str = "agent",
        top_k: int = 5,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/eval",
            json={"mode": mode, "top_k": top_k, "cases": cases},
        )

    def compare(self, document_id_a: str, document_id_b: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/compare",
            json={"document_id_a": document_id_a, "document_id_b": document_id_b},
        )

    def graph_traverse(
        self,
        start_entity_query: str,
        *,
        max_hops: int = 3,
        corpus_id: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/graph/traverse",
            json={
                "start_entity_query": start_entity_query,
                "max_hops": max_hops,
                "corpus_id": corpus_id,
                "document_id": document_id,
            },
        )

    def contradictions(
        self,
        *,
        document_id_a: str | None = None,
        document_id_b: str | None = None,
        corpus_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/contradictions",
            json={
                "document_id_a": document_id_a,
                "document_id_b": document_id_b,
                "corpus_id": corpus_id,
            },
        )

    def list_templates(self) -> dict[str, Any]:
        return self._request("GET", "/v1/templates")

    def extract_template(
        self,
        template_id: str,
        *,
        document_id: str | None = None,
        corpus_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/extract",
            json={"template_id": template_id, "document_id": document_id, "corpus_id": corpus_id},
        )

    def register_webhook(self, url: str, events: list[str] | None = None) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/webhooks",
            json={"url": url, "events": events or ["ingest.completed"]},
        )

    def list_webhooks(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/webhooks")

    def translate(self, text: str, *, source_language: str = "auto", target_language: str = "en-IN") -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/translate",
            json={"text": text, "source_language": source_language, "target_language": target_language},
        )

    def query_stream(
        self,
        question: str,
        *,
        top_k: int = 5,
        max_steps: int = 8,
        document_id: str | None = None,
        corpus_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        import json

        payload: dict[str, Any] = {
            "query": question,
            "top_k": top_k,
            "max_steps": max_steps,
        }
        if document_id:
            payload["document_id"] = document_id
        if corpus_id:
            payload["corpus_id"] = corpus_id
        try:
            resp = self.session.post(
                self._url("/v1/query/stream"),
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise SanrachnaError(str(exc)) from exc

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            yield json.loads(line[6:])

    def build_rag_context(self, search_response: dict[str, Any], *, max_chars: int = 6000) -> str:
        parts: list[str] = []
        used = 0
        if search_response.get("graph_context"):
            block = f"## Knowledge graph\n{search_response['graph_context']}"
            parts.append(block)
            used += len(block)
        for hit in search_response.get("results", []):
            block = f"[{hit.get('chunk_id')} score={hit.get('score', 0):.3f}]\n{hit.get('text', '')}"
            if hit.get("bbox"):
                block += f"\n(bbox page={hit.get('page')} {hit['bbox']})"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)
