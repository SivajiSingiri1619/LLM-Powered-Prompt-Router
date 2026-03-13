from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        instructions: str,
        user_message: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str: ...


class OpenAIResponsesClient:
    def __init__(self, *, api_key: str | None, base_url: str | None = None) -> None:
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY or GROQ_API_KEY is required to use the live router service. Put it in .env or .env.example and restart the server."
            )

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        *,
        model: str,
        instructions: str,
        user_message: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        response = self._client.responses.create(
            model=model,
            instructions=instructions,
            input=user_message,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return (response.output_text or "").strip()
