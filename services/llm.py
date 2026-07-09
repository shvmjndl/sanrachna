"""LiteLLM proxy client with cost tracking, tools, and reranking."""

from __future__ import annotations

import json

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.cost_tracker import CostTracker


class LLMService:
    def __init__(self, settings: Settings, cost_tracker: CostTracker | None = None):
        self._settings = settings
        self._cost = cost_tracker
        self._client = AsyncOpenAI(
            base_url=f"{settings.litellm_base_url}/v1",
            api_key=settings.litellm_api_key,
        )

    def _log_usage(
        self,
        model: str,
        operation: str,
        usage,
        job_id: str | None,
        document_id: str | None,
    ) -> None:
        if self._cost and usage:
            self._cost.log_litellm(
                job_id=job_id,
                document_id=document_id,
                model=model,
                operation=operation,
                input_tokens=usage.prompt_tokens,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        operation: str = "chat",
        job_id: str | None = None,
        document_id: str | None = None,
        temperature: float = 0.1,
        response_format: dict | None = None,
    ) -> tuple[str, dict]:
        model = model or self._settings.litellm_chat_model
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        self._log_usage(model, operation, response.usage, job_id, document_id)
        return content, usage

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        operation: str = "agent_tool_call",
        job_id: str | None = None,
        document_id: str | None = None,
        tool_choice: str | dict = "auto",
    ) -> dict:
        model = model or self._settings.litellm_chat_model
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.1,
        )
        self._log_usage(model, operation, response.usage, job_id, document_id)

        msg = response.choices[0].message
        result: dict = {"content": msg.content}

        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return result

    async def embed(
        self,
        text: str,
        *,
        operation: str = "embedding",
        job_id: str | None = None,
        document_id: str | None = None,
    ) -> tuple[list[float], dict]:
        model = self._settings.litellm_embedding_model
        response = await self._client.embeddings.create(model=model, input=text)
        embedding = response.data[0].embedding
        usage = {"input_tokens": response.usage.prompt_tokens if response.usage else 0}
        self._log_usage(model, operation, response.usage, job_id, document_id)
        return embedding, usage

    async def embed_batch(
        self,
        texts: list[str],
        *,
        operation: str = "embedding_batch",
        job_id: str | None = None,
        document_id: str | None = None,
        batch_size: int = 50,
    ) -> tuple[list[list[float]], dict]:
        all_embeddings: list[list[float]] = []
        total_tokens = 0
        model = self._settings.litellm_embedding_model

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._client.embeddings.create(model=model, input=batch)
            all_embeddings.extend([d.embedding for d in response.data])
            if response.usage:
                total_tokens += response.usage.prompt_tokens

        if self._cost:
            self._cost.log_litellm(
                job_id=job_id,
                document_id=document_id,
                model=model,
                operation=operation,
                input_tokens=total_tokens,
                output_tokens=0,
            )

        return all_embeddings, {"input_tokens": total_tokens}
