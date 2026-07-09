"""LangChain integration — optional, uses HTTP client when langchain not installed."""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.tools import tool

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    LCDocument = None
    BaseRetriever = object
    tool = lambda fn: fn


class SanrachnaRetriever(BaseRetriever if _HAS_LANGCHAIN else object):
    """Retriever that calls Sanrachna hybrid search API."""

    def __init__(self, client, top_k: int = 5, document_id: str | None = None):
        if _HAS_LANGCHAIN:
            super().__init__()
        self._client = client
        self._top_k = top_k
        self._document_id = document_id

    def _get_relevant_documents(self, query: str) -> list:
        if not _HAS_LANGCHAIN:
            raise ImportError("Install langchain-core: pip install langchain-core")
        data = self._client.search(query, top_k=self._top_k, document_id=self._document_id)
        docs = []
        for hit in data.get("results", []):
            docs.append(
                LCDocument(
                    page_content=hit.get("text", ""),
                    metadata={
                        "chunk_id": hit.get("chunk_id"),
                        "document_id": hit.get("document_id"),
                        "score": hit.get("score"),
                        "page": hit.get("page"),
                        "bbox": hit.get("bbox"),
                    },
                )
            )
        return docs

    async def _aget_relevant_documents(self, query: str) -> list:
        return self._get_relevant_documents(query)


def build_sanrachna_tools(client) -> list:
    """Return LangChain tools wrapping Sanrachna API."""

    if not _HAS_LANGCHAIN:
        return []

    @tool
    def sanrachna_search(query: str) -> str:
        """Search indexed documents using hybrid retrieval."""
        data = client.search(query, top_k=5)
        return client.build_rag_context(data)

    @tool
    def sanrachna_ask(query: str) -> str:
        """Ask the Sanrachna agent a question with citations."""
        data = client.query(query)
        return data.get("answer", "")

    @tool
    def sanrachna_extract(query: str, template_id: str = "financial_summary") -> str:
        """Extract structured fields using a schema template."""
        import json

        fields = client._request("GET", f"/v1/templates/{template_id}")
        data = client.extract(fields.get("fields", []))
        return json.dumps(data, ensure_ascii=False)

    return [sanrachna_search, sanrachna_ask, sanrachna_extract]
