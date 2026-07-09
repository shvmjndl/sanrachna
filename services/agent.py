"""Agent loop with prefetch RAG + tool use."""

from __future__ import annotations

import json

from app.core.config import Settings
from app.schemas.knowledge import AgentQueryResponse, AgentStep, SearchResult
from pipeline.indexing import HybridIndexer
from services.agent_tools import AGENT_TOOLS, SYSTEM_PROMPT, ToolExecutor
from services.confidence import assess_evidence
from services.entity_resolution import EntityResolver
from services.graph_service import GraphService
from services.llm import LLMService
from services.multilingual import MultilingualQueryService
from services.scope import ResolvedScope, ScopeResolver

_NO_EVIDENCE_PHRASES = (
    "couldn't find",
    "could not find",
    "no information",
    "not find any",
    "do not contain",
    "does not contain",
    "please provide",
    "please let me know",
)


class DocumentAgent:
    def __init__(
        self,
        settings: Settings,
        llm: LLMService,
        indexer: HybridIndexer,
        graph: GraphService,
        entity_resolver: EntityResolver,
        session_factory,
    ):
        self._settings = settings
        self._llm = llm
        self._indexer = indexer
        self._graph = graph
        self._scope = ScopeResolver(session_factory)
        self._session = session_factory
        self._executor = ToolExecutor(
            settings, llm, indexer, graph, entity_resolver, session_factory
        )
        self._multilingual = MultilingualQueryService(settings)

    def _prepare_query(self, query: str) -> tuple[str, str, dict]:
        meta = self._multilingual.preprocess(query)
        return meta["search_query"], query, meta

    async def _prefetch(
        self,
        query: str,
        scope: ResolvedScope,
        top_k: int,
    ) -> tuple[list[SearchResult], list[dict]]:
        prefetched = await self._indexer.search(
            query=query,
            top_k=top_k,
            document_id=scope.document_id,
            corpus_id=scope.corpus_id,
            hybrid=True,
            rerank=True,
            expand_parent=False,
        )
        tables = self._graph.search_tables(
            query,
            corpus_id=scope.corpus_id,
            document_id=scope.document_id,
            limit=top_k,
        )
        return prefetched, tables

    def _scope_message(self, scope: ResolvedScope) -> str:
        if scope.mode == "document":
            return f"Using document_id={scope.document_id}."
        if scope.mode == "corpus":
            return f"Using corpus_id={scope.corpus_id}."
        if scope.mode == "single_corpus":
            return f"Single indexed document: {scope.document_id}."
        if scope.mode == "latest":
            return f"Using most recently ingested document: {scope.document_id}."
        return "Searching across all indexed documents."

    def _answer_lacks_evidence(self, answer: str, prefetched: list[SearchResult]) -> bool:
        if not answer:
            return True
        lower = answer.lower()
        return any(phrase in lower for phrase in _NO_EVIDENCE_PHRASES) and bool(prefetched)

    async def run(
        self,
        query: str,
        document_id: str | None = None,
        corpus_id: str | None = None,
        top_k: int = 5,
        max_steps: int = 8,
    ) -> AgentQueryResponse:
        search_query, original_query, lang_meta = self._prepare_query(query)
        scope = self._scope.resolve(document_id=document_id, corpus_id=corpus_id)
        self._executor.set_context(scope.document_id, scope.corpus_id)
        steps: list[AgentStep] = []

        prefetched, prefetched_tables = await self._prefetch(search_query, scope, top_k)
        self._executor.citations.extend(prefetched)
        self._executor.found_tables.extend(prefetched_tables)

        table_block = "\n\n".join(
            f"[table {t.get('table_id')}] headers={t.get('headers')} rows={t.get('rows')}"
            for t in prefetched_tables
        ) or "No matching structured rows found."

        context_block = "\n\n".join(
            f"[{c.chunk_id}] (score={c.score:.3f}): {c.text[:800]}"
            + (
                f"\n(parent context: {c.parent_text[:400]}...)"
                if c.parent_text and len(c.parent_text) > len(c.text)
                else ""
            )
            for c in prefetched
        ) or "No matching chunks found."

        system = f"""{SYSTEM_PROMPT}
{self._scope_message(scope)}

IMPORTANT: Use the prefetched evidence below and tools to answer.
When structured rows are available, use values from the row whose first column best matches the question.
Only attribute a value to a label when both appear in the same row or record.

## Prefetched search results
{context_block}

## Prefetched structured rows
{table_block}"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": original_query},
        ]

        answer = ""
        for step_i in range(max_steps):
            tool_choice = "required" if step_i == 0 and not prefetched else "auto"
            response = await self._llm.chat_with_tools(
                messages=messages,
                tools=AGENT_TOOLS,
                operation="agent_loop",
                tool_choice=tool_choice,
            )

            if response.get("tool_calls"):
                for tc in response["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
                    result = await self._executor.execute(fn_name, fn_args, top_k)

                    steps.append(
                        AgentStep(
                            tool=fn_name,
                            input=fn_args,
                            output_summary=str(result)[:300],
                        )
                    )

                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(result, ensure_ascii=False)[:4000],
                        }
                    )
            else:
                answer = response.get("content", "")
                break

        if self._answer_lacks_evidence(answer, prefetched):
            answer, _ = await self._llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Answer using ONLY the provided evidence. Cite chunk IDs.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Evidence:\n{context_block}\n\nStructured rows:\n{table_block}\n\n"
                            f"Question: {original_query}"
                        ),
                    },
                ],
                operation="agent_rag_fallback",
            )
            steps.append(
                AgentStep(tool="rag_fallback", input={"query": query}, output_summary=answer[:200])
            )

        if not answer and not prefetched:
            docs = await self._executor.execute("list_documents", {}, top_k)
            answer = (
                "No indexed evidence matched your query. "
                f"Known documents: {json.dumps(docs, ensure_ascii=False)[:1500]}"
            )
            steps.append(
                AgentStep(tool="list_documents", input={}, output_summary=answer[:200])
            )

        if not answer:
            answer, _ = await self._llm.chat(
                messages=messages
                + [{"role": "user", "content": "Provide your final answer based on the evidence gathered."}],
                operation="agent_final",
            )

        graph = None
        if self._executor.found_entities:
            _, _, graph = self._executor.graph.lookup_entities(
                query,
                corpus_id=scope.corpus_id,
                document_id=scope.document_id,
            )

        answer = self._multilingual.postprocess_answer(answer, lang_meta.get("answer_language"))

        citations = self._executor.citations[:top_k]
        evidence = assess_evidence(
            citations,
            tables=self._executor.found_tables[:5],
            answer=answer,
        )

        return AgentQueryResponse(
            answer=answer,
            citations=citations,
            entities=self._executor.found_entities[:10],
            graph=graph,
            tables=self._executor.found_tables[:5],
            steps=steps,
            evidence=evidence,
            scope={
                "document_id": scope.document_id,
                "corpus_id": scope.corpus_id,
                "mode": scope.mode,
                "query_language": lang_meta.get("query_language"),
                "translated_query": lang_meta.get("translated", False),
            },
        )

    async def run_stream(
        self,
        query: str,
        document_id: str | None = None,
        corpus_id: str | None = None,
        top_k: int = 5,
        max_steps: int = 8,
    ):
        yield json.dumps({"event": "start", "query": query})
        response = await self.run(
            query=query,
            document_id=document_id,
            corpus_id=corpus_id,
            top_k=top_k,
            max_steps=max_steps,
        )
        for step in response.steps:
            yield json.dumps({"event": "step", "tool": step.tool, "summary": step.output_summary})
        yield json.dumps(
            {
                "event": "answer",
                "answer": response.answer,
                "evidence": response.evidence.model_dump() if response.evidence else None,
                "citations": [c.model_dump() for c in response.citations],
                "scope": response.scope,
            }
        )
        yield json.dumps({"event": "done"})
