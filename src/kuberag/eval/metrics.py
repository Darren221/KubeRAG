from collections.abc import Sequence
from typing import Any, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, computed_field

from kuberag.generation.citations import VerifiedCitation


class CorrectnessVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 = completely wrong; 1.0 = fully correct match with the golden answer.",
    )
    rationale: str = Field(description="One-sentence justification for the score.")


_CORRECTNESS_SYSTEM_PROMPT = """You are a strict grader for a Kubernetes Q&A system.

Given a question, a reference (golden) answer, and a candidate answer, score how well \
the candidate addresses the question in terms of factual content. Ignore style and \
wording differences; focus on whether the same facts are conveyed.

Scoring rubric:
- 1.0  : candidate is fully correct and covers what the golden covers.
- 0.7-0.9 : mostly correct, missing only minor points.
- 0.4-0.6 : partial — some key facts present, others missing or wrong.
- 0.1-0.3 : contradictory or mostly wrong.
- 0.0  : completely wrong, unrelated, or empty.

Be strict but fair. Provide a one-sentence rationale."""


class AnswerCorrectness:
    def __init__(
        self, *, client: AsyncOpenAI, model: str = "gpt-4o-mini"
    ) -> None:
        self.client = client
        self.model = model

    async def score(
        self,
        question: str,
        golden_answer: str,
        predicted_answer: str,
    ) -> CorrectnessVerdict:
        messages = [
            {"role": "system", "content": _CORRECTNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Golden answer: {golden_answer}\n\n"
                    f"Candidate answer: {predicted_answer}"
                ),
            },
        ]
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=cast(Any, messages),
            response_format=CorrectnessVerdict,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return CorrectnessVerdict(
                score=0.0, rationale="model returned no parseable verdict"
            )
        return parsed


class FaithfulnessClaim(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim: str = Field(description="An atomic factual statement extracted from the answer.")
    supported: bool = Field(
        description="True iff the claim is supported by the retrieved chunks."
    )
    reason: str = Field(description="Brief justification.")


class FaithfulnessVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    claims: list[FaithfulnessClaim] = Field(
        description="One entry per atomic claim extracted from the answer."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score(self) -> float:
        if not self.claims:
            return 1.0
        supported = sum(1 for c in self.claims if c.supported)
        return supported / len(self.claims)


_FAITHFULNESS_SYSTEM_PROMPT = """You judge whether each factual claim in an answer is \
supported by the retrieved context passages.

1. Extract each atomic factual claim from the answer (one fact per claim, no compound statements).
2. For each claim, decide whether the retrieved passages support it.
3. Provide a brief reason for each judgment.

Be strict: a claim is supported only if the passages explicitly state it or directly \
imply it. Background knowledge does not count. Stylistic statements like "this is useful" \
or "in summary" are not factual claims — skip them."""


class Faithfulness:
    def __init__(
        self, *, client: AsyncOpenAI, model: str = "gpt-4o-mini"
    ) -> None:
        self.client = client
        self.model = model

    async def score(
        self, answer: str, chunk_texts: Sequence[str]
    ) -> FaithfulnessVerdict:
        formatted_chunks = "\n\n".join(
            f"[{i + 1}] {text}" for i, text in enumerate(chunk_texts)
        ) or "(no context passages provided)"

        messages = [
            {"role": "system", "content": _FAITHFULNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Answer:\n{answer}\n\n"
                    f"Retrieved passages:\n{formatted_chunks}"
                ),
            },
        ]
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=cast(Any, messages),
            response_format=FaithfulnessVerdict,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return FaithfulnessVerdict(
                claims=[
                    FaithfulnessClaim(
                        claim="(model returned no parseable verdict)",
                        supported=False,
                        reason="judge produced no output",
                    )
                ]
            )
        return parsed


def recall_at_k(
    retrieved_sources: Sequence[str],
    expected_source_files: Sequence[str],
    k: int,
) -> float:
    if k < 0:
        raise ValueError("k must be non-negative")
    if not expected_source_files:
        return 1.0
    if k == 0:
        return 0.0

    top_k = set(retrieved_sources[:k])
    matched = sum(
        1
        for expected in expected_source_files
        if any(expected in source for source in top_k)
    )
    return matched / len(expected_source_files)


def citation_accuracy(verifications: Sequence[VerifiedCitation]) -> float:
    if not verifications:
        return 1.0
    supported = sum(1 for v in verifications if v.supported)
    return supported / len(verifications)
