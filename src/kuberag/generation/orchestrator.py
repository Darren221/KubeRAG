from typing import Any, Literal, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from kuberag.generation.citations import (
    CitationError,
    CitationVerifier,
    VerifiedCitation,
    parse_citations,
)
from kuberag.generation.confidence import (
    ConfidenceBreakdown,
    ConfidenceWeights,
    retrieval_confidence,
    score_confidence,
)
from kuberag.generation.generator import GenerationError, Generator
from kuberag.retrieval.fusion import FusedHit


class CompletenessJudgment(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 if the answer doesn't address the question, 1.0 if fully addressed.",
    )
    reason: str = Field(description="One-sentence justification.")


_COMPLETENESS_SYSTEM_PROMPT = """You judge whether an answer fully addresses a question.

Score 1.0 if the answer fully covers what was asked. Score around 0.5 if the answer \
covers some but not all parts. Score 0.0 if the answer does not address the question at all.

Provide a one-sentence reason."""


class CompletenessJudge:
    def __init__(
        self, *, client: AsyncOpenAI, model: str = "gpt-4o-mini"
    ) -> None:
        self.client = client
        self.model = model

    async def score(self, question: str, answer: str) -> CompletenessJudgment:
        messages = [
            {"role": "system", "content": _COMPLETENESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nAnswer: {answer}",
            },
        ]
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=cast(Any, messages),
            response_format=CompletenessJudgment,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return CompletenessJudgment(
                score=0.5, reason="model returned no parseable verdict"
            )
        return parsed


class InsufficientAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["insufficient"] = "insufficient"
    reason: str
    retrieved_chunks: list[FusedHit] = Field(default_factory=list)
    suggested_documents: list[str] = Field(default_factory=list)
    generated_text: str | None = None


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["grounded"] = "grounded"
    text: str
    citations: list[VerifiedCitation]
    confidence: ConfidenceBreakdown
    retrieved_chunks: list[FusedHit]


AnswerResult = GroundedAnswer | InsufficientAnswer


def _unique_sources(chunks: list[FusedHit]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for c in chunks:
        if c.source not in seen:
            seen.add(c.source)
            ordered.append(c.source)
    return ordered


class GenerationOrchestrator:
    def __init__(
        self,
        *,
        generator: Generator,
        verifier: CitationVerifier,
        completeness_judge: CompletenessJudge,
        confidence_threshold: float = 0.4,
        weights: ConfidenceWeights | None = None,
    ) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.generator = generator
        self.verifier = verifier
        self.completeness_judge = completeness_judge
        self.confidence_threshold = confidence_threshold
        self.weights = weights

    async def answer(
        self,
        question: str,
        chunks: list[FusedHit],
    ) -> AnswerResult:
        if not question.strip():
            raise ValueError("question must not be empty")

        if not chunks:
            return InsufficientAnswer(reason="no chunks retrieved")

        retrieval_score = retrieval_confidence(chunks)
        if retrieval_score < self.confidence_threshold:
            return InsufficientAnswer(
                reason=(
                    f"retrieval confidence {retrieval_score:.2f} below threshold "
                    f"{self.confidence_threshold:.2f}"
                ),
                retrieved_chunks=chunks,
                suggested_documents=_unique_sources(chunks),
            )

        try:
            answer = await self.generator.generate(question, chunks)
        except GenerationError as e:
            return InsufficientAnswer(
                reason=f"generation failed: {e}",
                retrieved_chunks=chunks,
                suggested_documents=_unique_sources(chunks),
            )

        if answer.insufficient_context:
            return InsufficientAnswer(
                reason="model determined context insufficient",
                retrieved_chunks=chunks,
                suggested_documents=_unique_sources(chunks),
                generated_text=answer.text,
            )

        try:
            parsed = parse_citations(answer, chunks)
        except CitationError as e:
            return InsufficientAnswer(
                reason=f"citation parse failed: {e}",
                retrieved_chunks=chunks,
                suggested_documents=_unique_sources(chunks),
                generated_text=answer.text,
            )

        verifications = await self.verifier.verify_all(parsed)
        completeness = await self.completeness_judge.score(question, answer.text)
        confidence = score_confidence(
            chunks, verifications, completeness.score, weights=self.weights
        )

        if confidence.composite < self.confidence_threshold:
            return InsufficientAnswer(
                reason=(
                    f"composite confidence {confidence.composite:.2f} below threshold "
                    f"{self.confidence_threshold:.2f}"
                ),
                retrieved_chunks=chunks,
                suggested_documents=_unique_sources(chunks),
                generated_text=answer.text,
            )

        return GroundedAnswer(
            text=answer.text,
            citations=verifications,
            confidence=confidence,
            retrieved_chunks=chunks,
        )
