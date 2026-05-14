import math

import pytest

from kuberag.config import Settings
from kuberag.ingest.dedupe import cosine_similarity, is_duplicate, nearest_similarity


def test_cosine_identical_vectors() -> None:
    a = [1.0, 0.0, 0.0]
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_handles_unnormalized_vectors() -> None:
    # Same direction, different magnitudes → cosine = 1
    assert cosine_similarity([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)


def test_nearest_similarity_empty_existing() -> None:
    sim, idx = nearest_similarity([1.0, 0.0], [])
    assert math.isinf(sim) and sim < 0
    assert idx is None


def test_nearest_similarity_finds_closest_index() -> None:
    query = [1.0, 0.0, 0.0]
    existing = [
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    sim, idx = nearest_similarity(query, existing)
    assert idx == 1
    assert sim > 0.95


def test_is_duplicate_identical() -> None:
    assert is_duplicate([1.0, 0.0], [[1.0, 0.0]], threshold=0.95) is True


def test_is_duplicate_dissimilar() -> None:
    assert is_duplicate([1.0, 0.0], [[0.0, 1.0]], threshold=0.95) is False


def test_is_duplicate_at_exact_threshold() -> None:
    # Geometrically construct two vectors with cosine similarity = 0.95
    theta = math.acos(0.95)
    a = [1.0, 0.0]
    b = [math.cos(theta), math.sin(theta)]
    assert is_duplicate(a, [b], threshold=0.95) is True
    # Threshold just above → no longer duplicate
    assert is_duplicate(a, [b], threshold=0.951) is False


def test_is_duplicate_empty_existing() -> None:
    assert is_duplicate([1.0, 0.0], [], threshold=0.95) is False


def test_is_duplicate_finds_max_among_many() -> None:
    query = [1.0, 0.0, 0.0]
    existing = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.99, 0.1, 0.0],
    ]
    assert is_duplicate(query, existing, threshold=0.95) is True


def test_is_duplicate_default_threshold_is_0_95() -> None:
    assert is_duplicate([1.0, 0.0], [[1.0, 0.0]]) is True
    # Below 0.95 → kept
    theta = math.acos(0.9)
    b = [math.cos(theta), math.sin(theta)]
    assert is_duplicate([1.0, 0.0], [b]) is False


def test_threshold_from_settings_is_applied() -> None:
    settings = Settings(OPENAI_API_KEY="sk-test", dedupe_threshold=0.99)
    theta = math.acos(0.97)
    a = [1.0, 0.0]
    b = [math.cos(theta), math.sin(theta)]
    # 0.97 < 0.99 → kept under settings threshold
    assert is_duplicate(a, [b], threshold=settings.dedupe_threshold) is False
