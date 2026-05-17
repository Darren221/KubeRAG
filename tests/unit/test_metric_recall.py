import pytest

from kuberag.eval.metrics import recall_at_k


def test_all_expected_retrieved_yields_one() -> None:
    sources = ["docs/a.md", "docs/b.md", "docs/c.md"]
    expected = ["docs/a.md", "docs/b.md"]
    assert recall_at_k(sources, expected, k=5) == 1.0


def test_half_retrieved_yields_half() -> None:
    sources = ["docs/a.md", "docs/c.md"]
    expected = ["docs/a.md", "docs/b.md", "docs/c.md", "docs/d.md"]
    assert recall_at_k(sources, expected, k=5) == pytest.approx(2 / 4)


def test_none_retrieved_yields_zero() -> None:
    sources = ["docs/x.md", "docs/y.md"]
    expected = ["docs/a.md", "docs/b.md"]
    assert recall_at_k(sources, expected, k=5) == 0.0


def test_empty_expected_yields_one_vacuously() -> None:
    sources = ["docs/a.md"]
    expected: list[str] = []
    assert recall_at_k(sources, expected, k=5) == 1.0


def test_k_cuts_off_retrieval() -> None:
    sources = ["docs/x.md", "docs/y.md", "docs/a.md"]  # a.md is at rank 2
    expected = ["docs/a.md"]
    assert recall_at_k(sources, expected, k=2) == 0.0  # cut off before a.md
    assert recall_at_k(sources, expected, k=3) == 1.0  # a.md included


def test_k_larger_than_retrieved_uses_all_retrieved() -> None:
    sources = ["docs/a.md", "docs/b.md"]
    expected = ["docs/a.md", "docs/b.md"]
    assert recall_at_k(sources, expected, k=10) == 1.0


def test_k_zero_yields_zero() -> None:
    sources = ["docs/a.md"]
    expected = ["docs/a.md"]
    assert recall_at_k(sources, expected, k=0) == 0.0


def test_negative_k_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["docs/a.md"], ["docs/a.md"], k=-1)


def test_substring_match_for_absolute_paths() -> None:
    # In practice, ChromaDB stores absolute paths; golden set stores relative paths.
    # Substring matching handles the gap.
    sources = ["/tmp/clone/content/en/docs/concepts/workloads/pods/_index.md"]
    expected = ["content/en/docs/concepts/workloads/pods/_index.md"]
    assert recall_at_k(sources, expected, k=1) == 1.0


def test_duplicate_chunks_from_same_source_count_once() -> None:
    sources = ["docs/a.md", "docs/a.md", "docs/a.md"]
    expected = ["docs/a.md", "docs/b.md"]
    # Only 'a' was retrieved, even though it appeared 3 times
    assert recall_at_k(sources, expected, k=3) == 0.5


def test_duplicate_expected_files_count_independently() -> None:
    # If the golden lists the same file twice (unusual but valid), both count
    sources = ["docs/a.md"]
    expected = ["docs/a.md", "docs/a.md"]
    assert recall_at_k(sources, expected, k=1) == 1.0


def test_empty_retrieved_returns_zero_when_expected_nonempty() -> None:
    sources: list[str] = []
    expected = ["docs/a.md"]
    assert recall_at_k(sources, expected, k=5) == 0.0


def test_both_empty_returns_one_vacuously() -> None:
    assert recall_at_k([], [], k=5) == 1.0
