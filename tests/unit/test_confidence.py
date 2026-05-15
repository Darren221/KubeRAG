import pytest

from kuberag.generation.citations import VerifiedCitation
from kuberag.generation.confidence import (
    ConfidenceBreakdown,
    ConfidenceWeights,
    score_confidence,
)
from kuberag.retrieval.fusion import FusedHit


def make_fused_hit(*, rank: int = 0, rrf_score: float = 0.0164) -> FusedHit:
    return FusedHit(
        chunk_id=f"c{rank}",
        text=f"text {rank}",
        source=f"/test/c{rank}.md",
        section=None,
        chunking_strategy="fixed",
        rrf_score=rrf_score,
        rank=rank,
        dense_rank=rank,
        sparse_rank=rank,
    )


def make_verified(*, marker: int = 1, supported: bool = True) -> VerifiedCitation:
    return VerifiedCitation(
        marker=marker,
        claim_span=f"claim {marker}",
        chunk_id=f"c{marker}",
        source=f"/test/c{marker}.md",
        section=None,
        chunk_text=f"text {marker}",
        supported=supported,
        reason="ok",
    )


# --- composite math ---


def test_all_supported_high_retrieval_yields_high_composite() -> None:
    hits = [make_fused_hit(rank=i, rrf_score=0.016) for i in range(3)]
    verifications = [make_verified(marker=i + 1, supported=True) for i in range(3)]
    result = score_confidence(hits, verifications, completeness_score=0.9)
    assert result.composite >= 0.85


def test_one_unsupported_drops_citation_dimension() -> None:
    hits = [make_fused_hit(rank=i) for i in range(3)]
    all_supported = [make_verified(marker=i + 1, supported=True) for i in range(3)]
    one_bad = [
        make_verified(marker=1, supported=True),
        make_verified(marker=2, supported=False),
        make_verified(marker=3, supported=True),
    ]
    good = score_confidence(hits, all_supported, completeness_score=1.0)
    worse = score_confidence(hits, one_bad, completeness_score=1.0)
    assert worse.citation < good.citation
    assert worse.citation == pytest.approx(2 / 3, abs=1e-6)
    assert good.citation == pytest.approx(1.0, abs=1e-6)


def test_completeness_factored_into_composite() -> None:
    hits = [make_fused_hit()]
    verifications = [make_verified(supported=True)]
    high = score_confidence(hits, verifications, completeness_score=1.0)
    low = score_confidence(hits, verifications, completeness_score=0.0)
    assert high.composite > low.composite


def test_composite_clamped_to_unit_interval() -> None:
    hits = [make_fused_hit(rrf_score=99.0) for _ in range(5)]
    verifications = [make_verified() for _ in range(5)]
    result = score_confidence(hits, verifications, completeness_score=1.0)
    assert 0.0 <= result.composite <= 1.0


# --- empty inputs ---


def test_empty_hits_zero_retrieval() -> None:
    verifications = [make_verified()]
    result = score_confidence([], verifications, completeness_score=1.0)
    assert result.retrieval == 0.0


def test_empty_verifications_zero_citation() -> None:
    hits = [make_fused_hit()]
    result = score_confidence(hits, [], completeness_score=1.0)
    assert result.citation == 0.0


def test_all_empty_yields_zero_for_retrieval_and_citation() -> None:
    result = score_confidence([], [], completeness_score=0.0)
    assert result.retrieval == 0.0
    assert result.citation == 0.0
    assert result.composite == 0.0


# --- breakdown shape ---


def test_returns_typed_breakdown() -> None:
    hits = [make_fused_hit()]
    verifications = [make_verified()]
    result = score_confidence(hits, verifications, completeness_score=0.5)
    assert isinstance(result, ConfidenceBreakdown)


def test_completeness_passed_through_to_breakdown() -> None:
    hits = [make_fused_hit()]
    verifications = [make_verified()]
    result = score_confidence(hits, verifications, completeness_score=0.73)
    assert result.completeness == pytest.approx(0.73)


# --- validation ---


def test_completeness_must_be_in_unit_interval() -> None:
    with pytest.raises(ValueError):
        score_confidence([], [], completeness_score=1.5)
    with pytest.raises(ValueError):
        score_confidence([], [], completeness_score=-0.1)


def test_custom_weights_applied() -> None:
    hits = [make_fused_hit(rrf_score=0.016)]
    verifications = [make_verified(supported=True)]
    # All weight on retrieval
    weights = ConfidenceWeights(retrieval=1.0, citation=0.0, completeness=0.0)
    result = score_confidence(
        hits, verifications, completeness_score=0.0, weights=weights
    )
    assert result.composite == pytest.approx(result.retrieval)


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        ConfidenceWeights(retrieval=0.5, citation=0.5, completeness=0.5)


def test_weights_reject_negative() -> None:
    with pytest.raises(ValueError):
        ConfidenceWeights(retrieval=-0.1, citation=0.6, completeness=0.5)


# --- proportional citation ---


def test_two_of_three_supported_gives_two_thirds() -> None:
    verifications = [
        make_verified(marker=1, supported=True),
        make_verified(marker=2, supported=True),
        make_verified(marker=3, supported=False),
    ]
    result = score_confidence([make_fused_hit()], verifications, completeness_score=1.0)
    assert result.citation == pytest.approx(2 / 3, abs=1e-6)


def test_all_unsupported_gives_zero_citation() -> None:
    verifications = [make_verified(marker=i + 1, supported=False) for i in range(3)]
    result = score_confidence([make_fused_hit()], verifications, completeness_score=1.0)
    assert result.citation == 0.0
