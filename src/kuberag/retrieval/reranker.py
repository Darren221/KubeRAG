from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from kuberag.retrieval.fusion import FusedHit

_SYSTEM_PROMPT = """You are a relevance judge for a retrieval system.

Given a question and a numbered list of candidate text passages, rank the
passages by how well each one answers the question.

For each relevant candidate, return its 0-indexed position from the input
list along with a one-sentence justification. List most relevant first.
Exclude any candidate that is unrelated to the question."""


class RerankedIndex(BaseModel):
    index: int = Field(ge=0, description="0-indexed position in the input list")
    reason: str = Field(description="One-sentence justification")


class RerankResponse(BaseModel):
    ranked: list[RerankedIndex] = Field(
        description="Candidates ordered by relevance, most relevant first"
    )


class Reranker:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        top_n: int = 5,
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self.client = client
        self.model = model
        self.top_n = top_n

    async def rerank(
        self,
        query: str,
        candidates: list[FusedHit],
        *,
        top_n: int | None = None,
    ) -> list[FusedHit]:
        if not candidates:
            return []
        effective_top_n = top_n if top_n is not None else self.top_n
        if effective_top_n <= 0:
            raise ValueError("top_n must be positive")

        user_prompt = self._build_user_prompt(query, candidates)
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=RerankResponse,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return []

        valid_indexes: list[int] = []
        seen: set[int] = set()
        for entry in parsed.ranked:
            if entry.index < 0 or entry.index >= len(candidates):
                continue
            if entry.index in seen:
                continue
            seen.add(entry.index)
            valid_indexes.append(entry.index)

        top = valid_indexes[:effective_top_n]
        return [
            candidates[idx].model_copy(update={"rank": new_rank})
            for new_rank, idx in enumerate(top)
        ]

    @staticmethod
    def _build_user_prompt(query: str, candidates: list[FusedHit]) -> str:
        formatted = "\n\n".join(
            f"[{i}] (source: {c.source})\n{c.text}"
            for i, c in enumerate(candidates)
        )
        return (
            f"Question: {query}\n\n"
            f"Candidates:\n{formatted}\n\n"
            "Rank the candidates by relevance to the question."
        )
