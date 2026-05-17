import pytest

from kuberag.eval.metrics import citation_accuracy
from kuberag.generation.citations import VerifiedCitation


def _verified(marker: int = 1, supported: bool = True) -> VerifiedCitation:
    return VerifiedCitation(
        marker=marker,
        claim_span=f"claim {marker}",
        chunk_id=f"c{marker}",
        source=f"/test/c{marker}.md",
        section=None,
        chunk_text="...",
        supported=supported,
        reason="ok",
    )


def test_all_supported_yields_one() -> None:
    citations = [_verified(marker=i + 1, supported=True) for i in range(5)]
    assert citation_accuracy(citations) == 1.0


def test_four_of_five_supported_yields_point_eight() -> None:
    citations = [
        _verified(marker=1, supported=True),
        _verified(marker=2, supported=True),
        _verified(marker=3, supported=True),
        _verified(marker=4, supported=True),
        _verified(marker=5, supported=False),
    ]
    assert citation_accuracy(citations) == pytest.approx(0.8)


def test_half_supported_yields_half() -> None:
    citations = [
        _verified(marker=1, supported=True),
        _verified(marker=2, supported=False),
    ]
    assert citation_accuracy(citations) == 0.5


def test_none_supported_yields_zero() -> None:
    citations = [_verified(marker=i + 1, supported=False) for i in range(3)]
    assert citation_accuracy(citations) == 0.0


def test_empty_citations_yields_one_vacuously() -> None:
    # No citations (e.g., insufficient_context refusal) = no unsupported citations
    assert citation_accuracy([]) == 1.0


def test_single_supported_yields_one() -> None:
    assert citation_accuracy([_verified(supported=True)]) == 1.0


def test_single_unsupported_yields_zero() -> None:
    assert citation_accuracy([_verified(supported=False)]) == 0.0
