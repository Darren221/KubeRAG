from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kuberag.generation.citations import VerifiedCitation
from kuberag.retrieval.fusion import FusedHit

# Theoretical max RRF score with k_constant=60 and a hit at rank 0 in both lists.
# Used to normalize rrf_score into a 0-1 retrieval-confidence value.
_RRF_NORMALIZER = 60.0
_TOP_K_FOR_RETRIEVAL_SCORE = 3


class ConfidenceWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval: float = Field(default=0.4, ge=0.0, le=1.0)
    citation: float = Field(default=0.4, ge=0.0, le=1.0)
    completeness: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sums_to_one(self) -> Self:
        total = self.retrieval + self.citation + self.completeness
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"weights must sum to 1.0 (got {total:.4f})")
        return self


class ConfidenceBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval: float = Field(ge=0.0, le=1.0)
    citation: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    composite: float = Field(ge=0.0, le=1.0)


def _retrieval_confidence(hits: list[FusedHit]) -> float:
    if not hits:
        return 0.0
    top = hits[:_TOP_K_FOR_RETRIEVAL_SCORE]
    avg = sum(h.rrf_score for h in top) / len(top)
    return min(1.0, max(0.0, avg * _RRF_NORMALIZER))


def _citation_confidence(verifications: list[VerifiedCitation]) -> float:
    if not verifications:
        return 0.0
    supported = sum(1 for v in verifications if v.supported)
    return supported / len(verifications)


def score_confidence(
    retrieval_hits: list[FusedHit],
    verifications: list[VerifiedCitation],
    completeness_score: float,
    *,
    weights: ConfidenceWeights | None = None,
) -> ConfidenceBreakdown:
    if not (0.0 <= completeness_score <= 1.0):
        raise ValueError("completeness_score must be in [0, 1]")

    w = weights or ConfidenceWeights()
    retrieval = _retrieval_confidence(retrieval_hits)
    citation = _citation_confidence(verifications)
    composite = (
        w.retrieval * retrieval
        + w.citation * citation
        + w.completeness * completeness_score
    )
    return ConfidenceBreakdown(
        retrieval=retrieval,
        citation=citation,
        completeness=completeness_score,
        composite=min(1.0, max(0.0, composite)),
    )
