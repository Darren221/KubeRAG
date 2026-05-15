import pytest

from kuberag.retrieval.fusion import FusedHit, rrf
from kuberag.stores import Hit


def make_hit(
    chunk_id: str,
    rank: int,
    *,
    text: str | None = None,
    source: str | None = None,
    section: str | None = None,
    score: float = 1.0,
) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        text=text or f"text for {chunk_id}",
        source=source or f"/test/{chunk_id}.md",
        section=section,
        chunking_strategy="fixed",
        score=score,
        rank=rank,
    )


def test_empty_inputs_return_empty() -> None:
    assert rrf([], []) == []


def test_dense_only_passthrough() -> None:
    dense = [make_hit("a", 0), make_hit("b", 1), make_hit("c", 2)]
    result = rrf(dense, [])
    assert [h.chunk_id for h in result] == ["a", "b", "c"]
    assert all(h.sparse_rank is None for h in result)
    assert [h.dense_rank for h in result] == [0, 1, 2]


def test_sparse_only_passthrough() -> None:
    sparse = [make_hit("a", 0), make_hit("b", 1)]
    result = rrf([], sparse)
    assert [h.chunk_id for h in result] == ["a", "b"]
    assert all(h.dense_rank is None for h in result)
    assert [h.sparse_rank for h in result] == [0, 1]


def test_chunk_in_both_lists_outscores_either_alone() -> None:
    dense = [make_hit("a", 0), make_hit("b", 1)]
    sparse = [make_hit("a", 0), make_hit("c", 1)]
    result = rrf(dense, sparse, dense_weight=0.7)

    top = result[0]
    assert top.chunk_id == "a"
    assert top.dense_rank == 0
    assert top.sparse_rank == 0
    # And it beats both b (dense-only) and c (sparse-only)
    a_score = next(h.rrf_score for h in result if h.chunk_id == "a")
    b_score = next(h.rrf_score for h in result if h.chunk_id == "b")
    c_score = next(h.rrf_score for h in result if h.chunk_id == "c")
    assert a_score > b_score
    assert a_score > c_score


def test_higher_rank_outscores_lower_rank() -> None:
    dense = [make_hit("a", 0), make_hit("b", 1), make_hit("c", 2)]
    result = rrf(dense, [])
    assert result[0].rrf_score > result[1].rrf_score > result[2].rrf_score


def test_dense_weight_one_uses_only_dense() -> None:
    dense = [make_hit("a", 0)]
    sparse = [make_hit("b", 0)]
    result = rrf(dense, sparse, dense_weight=1.0)
    by_id = {h.chunk_id: h for h in result}
    assert by_id["a"].rrf_score > 0
    assert by_id["b"].rrf_score == 0


def test_dense_weight_zero_uses_only_sparse() -> None:
    dense = [make_hit("a", 0)]
    sparse = [make_hit("b", 0)]
    result = rrf(dense, sparse, dense_weight=0.0)
    by_id = {h.chunk_id: h for h in result}
    assert by_id["a"].rrf_score == 0
    assert by_id["b"].rrf_score > 0


def test_dense_weight_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        rrf([make_hit("a", 0)], [], dense_weight=1.5)
    with pytest.raises(ValueError):
        rrf([make_hit("a", 0)], [], dense_weight=-0.1)


def test_k_constant_must_be_positive() -> None:
    with pytest.raises(ValueError):
        rrf([make_hit("a", 0)], [], k_constant=0)


def test_score_formula_for_top_dense_hit() -> None:
    dense = [make_hit("a", 0)]
    result = rrf(dense, [], dense_weight=0.7, k_constant=60)
    # Formula: dense_weight / (k_constant + rank+1) = 0.7 / 61
    assert result[0].rrf_score == pytest.approx(0.7 / 61)


def test_smaller_k_constant_amplifies_top_rank() -> None:
    dense = [make_hit("a", 0)]
    s60 = rrf(dense, [], k_constant=60)[0].rrf_score
    s10 = rrf(dense, [], k_constant=10)[0].rrf_score
    assert s10 > s60


def test_output_sorted_by_score_descending() -> None:
    dense = [make_hit("a", 0), make_hit("b", 1), make_hit("c", 2)]
    result = rrf(dense, [])
    scores = [h.rrf_score for h in result]
    assert scores == sorted(scores, reverse=True)


def test_final_rank_is_zero_indexed_sequential() -> None:
    dense = [make_hit("a", 0), make_hit("b", 1)]
    sparse = [make_hit("c", 0)]
    result = rrf(dense, sparse)
    assert [h.rank for h in result] == list(range(len(result)))


def test_metadata_preserved_from_hits() -> None:
    dense = [make_hit("a", 0, source="/docs/intro.md", section="Pods", text="pods are units")]
    result = rrf(dense, [])
    assert result[0].source == "/docs/intro.md"
    assert result[0].section == "Pods"
    assert result[0].text == "pods are units"


def test_metadata_falls_back_to_sparse_if_chunk_only_in_sparse() -> None:
    sparse = [make_hit("a", 0, source="/sparse/only.md")]
    result = rrf([], sparse)
    assert result[0].source == "/sparse/only.md"


def test_returns_fused_hit_models() -> None:
    result = rrf([make_hit("a", 0)], [])
    assert all(isinstance(h, FusedHit) for h in result)


def test_disjoint_lists_produce_union() -> None:
    dense = [make_hit("a", 0), make_hit("b", 1)]
    sparse = [make_hit("c", 0), make_hit("d", 1)]
    result = rrf(dense, sparse)
    assert {h.chunk_id for h in result} == {"a", "b", "c", "d"}
