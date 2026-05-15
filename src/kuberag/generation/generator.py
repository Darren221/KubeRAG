from typing import Any, cast

from openai import AsyncOpenAI

from kuberag.generation.models import Answer
from kuberag.generation.prompts import build_prompt
from kuberag.retrieval.fusion import FusedHit


class GenerationError(RuntimeError):
    """Raised when the model returns no parseable answer or contradictory output."""


class Generator:
    def __init__(self, *, client: AsyncOpenAI, model: str = "gpt-4o") -> None:
        self.client = client
        self.model = model

    async def generate(self, question: str, chunks: list[FusedHit]) -> Answer:
        if not question.strip():
            raise ValueError("question must not be empty")

        messages = build_prompt(question, chunks)
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=cast(Any, messages),
            response_format=Answer,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise GenerationError("model returned no parseable answer")
        if not parsed.insufficient_context and not parsed.citations:
            raise GenerationError(
                "model claimed grounded answer but produced no citations"
            )
        return parsed
