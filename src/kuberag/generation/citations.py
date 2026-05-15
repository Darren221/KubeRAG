import asyncio
import re
from typing import Any, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from kuberag.generation.models import Answer
from kuberag.retrieval.fusion import FusedHit

_MARKER_RE = re.compile(r"\[(\d+)\]")


class CitationError(ValueError):
    """Raised when a citation marker does not reference a valid chunk."""


class ParsedCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    marker: int = Field(ge=1)
    claim_span: str
    chunk_id: str
    source: str
    section: str | None = None
    chunk_text: str


def find_text_markers(text: str) -> set[int]:
    return {int(m.group(1)) for m in _MARKER_RE.finditer(text)}


def parse_citations(answer: Answer, chunks: list[FusedHit]) -> list[ParsedCitation]:
    text_markers = find_text_markers(answer.text)
    structured_markers = {c.marker for c in answer.citations}
    all_markers = text_markers | structured_markers

    if all_markers and not chunks:
        raise CitationError(
            "answer contains citation markers but no chunks were supplied"
        )

    for marker in all_markers:
        if marker < 1 or marker > len(chunks):
            raise CitationError(
                f"citation marker [{marker}] does not map to any chunk "
                f"(supplied {len(chunks)} chunks)"
            )

    result: list[ParsedCitation] = []
    for citation in answer.citations:
        chunk = chunks[citation.marker - 1]
        result.append(
            ParsedCitation(
                marker=citation.marker,
                claim_span=citation.claim_span,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                section=chunk.section,
                chunk_text=chunk.text,
            )
        )
    return result


class VerificationResponse(BaseModel):
    supported: bool = Field(
        description="True if the passage supports the claim, false otherwise."
    )
    reason: str = Field(description="One-sentence justification.")


class VerifiedCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    marker: int = Field(ge=1)
    claim_span: str
    chunk_id: str
    source: str
    section: str | None = None
    chunk_text: str
    supported: bool
    reason: str


_VERIFIER_SYSTEM_PROMPT = """You judge whether a passage supports a claim.

Read the passage and the claim. Decide whether the passage contains information \
that supports the claim, in part or in full.

Return supported=true if the passage clearly supports the claim. \
Return supported=false otherwise.

Be strict: if the passage is on the same topic but does not specifically support \
the claim, return supported=false. Provide a one-sentence reason in either case."""


class CitationVerifier:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        max_concurrency: int = 5,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.client = client
        self.model = model
        self.max_concurrency = max_concurrency

    async def verify(self, claim: str, chunk_text: str) -> VerificationResponse:
        messages = [
            {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Claim: {claim}\n\nPassage: {chunk_text}",
            },
        ]
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=cast(Any, messages),
            response_format=VerificationResponse,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return VerificationResponse(
                supported=False, reason="model returned no parseable verdict"
            )
        return parsed

    async def verify_all(
        self, citations: list[ParsedCitation]
    ) -> list[VerifiedCitation]:
        if not citations:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def verify_one(citation: ParsedCitation) -> VerifiedCitation:
            async with semaphore:
                verdict = await self.verify(citation.claim_span, citation.chunk_text)
            return VerifiedCitation(
                marker=citation.marker,
                claim_span=citation.claim_span,
                chunk_id=citation.chunk_id,
                source=citation.source,
                section=citation.section,
                chunk_text=citation.chunk_text,
                supported=verdict.supported,
                reason=verdict.reason,
            )

        return await asyncio.gather(*(verify_one(c) for c in citations))
