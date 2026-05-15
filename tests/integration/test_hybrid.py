from typing import Any
from unittest.mock import AsyncMock

import pytest

from kuberag.retrieval.fusion import FusedHit
from kuberag.retrieval.hybrid import HybridSearch
from kuberag.stores import Hit

pytestmark = pytest.mark.integration


def make_hit(chunk_id: str, rank: int, *, score: float = 1.0) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        source=f"/test/{chunk_id}.md",
        section=None,
        chunking_strategy="fixed",
        score=score,
        rank=rank,
    )


def make_fused(chunk_id: str, rank: int) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        source=f"/test/{chunk_id}.md",
        section=None,
        chunking_strategy="fixed",
        rrf_score=0.5,
        rank=rank,
        dense_rank=rank,
        sparse_rank=rank,
    )


@pytest.fixture
def mock_dense() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_sparse() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_reranker() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def hybrid(
    mock_dense: AsyncMock, mock_sparse: AsyncMock, mock_reranker: AsyncMock
) -> HybridSearch:
    return HybridSearch(
        dense=mock_dense, sparse=mock_sparse, reranker=mock_reranker
    )


async def test_dense_only_skips_sparse(
    hybrid: HybridSearch,
    mock_dense: AsyncMock,
    mock_sparse: AsyncMock,
    mock_reranker: AsyncMock,
) -> None:
    mock_dense.retrieve.return_value = [make_hit("a", 0)]
    mock_reranker.rerank.return_value = [make_fused("a", 0)]
    await hybrid.search("query", dense_only=True)
    mock_dense.retrieve.assert_called_once()
    mock_sparse.retrieve.assert_not_called()


async def test_hybrid_calls_both_retrievers(
    hybrid: HybridSearch,
    mock_dense: AsyncMock,
    mock_sparse: AsyncMock,
    mock_reranker: AsyncMock,
) -> None:
    mock_dense.retrieve.return_value = [make_hit("a", 0)]
    mock_sparse.retrieve.return_value = [make_hit("b", 0)]
    mock_reranker.rerank.return_value = [make_fused("a", 0)]
    await hybrid.search("query")
    mock_dense.retrieve.assert_called_once()
    mock_sparse.retrieve.assert_called_once()


async def test_k_propagates_to_both_retrievers(
    hybrid: HybridSearch,
    mock_dense: AsyncMock,
    mock_sparse: AsyncMock,
    mock_reranker: AsyncMock,
) -> None:
    mock_dense.retrieve.return_value = []
    mock_sparse.retrieve.return_value = []
    mock_reranker.rerank.return_value = []
    await hybrid.search("query", k=15)
    assert mock_dense.retrieve.call_args.kwargs["k"] == 15
    assert mock_sparse.retrieve.call_args.kwargs["k"] == 15


async def test_top_n_propagates_to_reranker(
    hybrid: HybridSearch,
    mock_dense: AsyncMock,
    mock_sparse: AsyncMock,
    mock_reranker: AsyncMock,
) -> None:
    mock_dense.retrieve.return_value = [make_hit("a", 0)]
    mock_sparse.retrieve.return_value = []
    mock_reranker.rerank.return_value = [make_fused("a", 0)]
    await hybrid.search("query", top_n=3)
    assert mock_reranker.rerank.call_args.kwargs["top_n"] == 3


async def test_empty_query_returns_empty_without_calls(
    hybrid: HybridSearch,
    mock_dense: AsyncMock,
    mock_sparse: AsyncMock,
    mock_reranker: AsyncMock,
) -> None:
    result = await hybrid.search("")
    assert result == []
    mock_dense.retrieve.assert_not_called()
    mock_sparse.retrieve.assert_not_called()
    mock_reranker.rerank.assert_not_called()


async def test_dense_only_path_passes_fused_hits_to_reranker(
    hybrid: HybridSearch,
    mock_dense: AsyncMock,
    mock_sparse: AsyncMock,
    mock_reranker: AsyncMock,
) -> None:
    mock_dense.retrieve.return_value = [
        make_hit("a", 0, score=0.95),
        make_hit("b", 1, score=0.80),
    ]
    captured: list[Any] = []

    async def capture(query: str, candidates: list[FusedHit], **kwargs: Any) -> list[FusedHit]:
        captured.extend(candidates)
        return candidates

    mock_reranker.rerank.side_effect = capture
    await hybrid.search("query", dense_only=True)

    assert len(captured) == 2
    assert all(isinstance(c, FusedHit) for c in captured)
    assert captured[0].chunk_id == "a"
    assert captured[0].dense_rank == 0
    assert captured[0].sparse_rank is None


async def test_hybrid_path_passes_rrf_output_to_reranker(
    hybrid: HybridSearch,
    mock_dense: AsyncMock,
    mock_sparse: AsyncMock,
    mock_reranker: AsyncMock,
) -> None:
    mock_dense.retrieve.return_value = [make_hit("a", 0), make_hit("b", 1)]
    mock_sparse.retrieve.return_value = [make_hit("c", 0), make_hit("a", 1)]
    captured: list[FusedHit] = []

    async def capture(query: str, candidates: list[FusedHit], **kwargs: Any) -> list[FusedHit]:
        captured.extend(candidates)
        return candidates

    mock_reranker.rerank.side_effect = capture
    await hybrid.search("query", dense_only=False)

    ids = {c.chunk_id for c in captured}
    assert ids == {"a", "b", "c"}
    a = next(c for c in captured if c.chunk_id == "a")
    assert a.dense_rank == 0
    assert a.sparse_rank == 1


async def test_returns_reranker_output(
    hybrid: HybridSearch,
    mock_dense: AsyncMock,
    mock_sparse: AsyncMock,
    mock_reranker: AsyncMock,
) -> None:
    mock_dense.retrieve.return_value = [make_hit("a", 0)]
    mock_sparse.retrieve.return_value = []
    final = [make_fused("a", 0)]
    mock_reranker.rerank.return_value = final
    result = await hybrid.search("query")
    assert result == final


async def test_dense_only_and_hybrid_differ_on_keyword_query(
    hybrid: HybridSearch,
    mock_dense: AsyncMock,
    mock_sparse: AsyncMock,
    mock_reranker: AsyncMock,
) -> None:
    # Dense retrieval favors chunk "semantic-match" highly;
    # Sparse retrieval finds an exact keyword match in "keyword-match".
    mock_dense.retrieve.return_value = [
        make_hit("semantic-match", 0),
        make_hit("keyword-match", 5),
    ]
    mock_sparse.retrieve.return_value = [
        make_hit("keyword-match", 0),
        make_hit("noise", 1),
    ]

    captured_calls: list[list[str]] = []

    async def capture(query: str, candidates: list[FusedHit], **kwargs: Any) -> list[FusedHit]:
        captured_calls.append([c.chunk_id for c in candidates])
        return candidates

    mock_reranker.rerank.side_effect = capture

    await hybrid.search("query", dense_only=True)
    await hybrid.search("query", dense_only=False)

    # The two paths feed the reranker different candidate sets
    assert captured_calls[0] != captured_calls[1]
    assert set(captured_calls[1]) > set(captured_calls[0])  # hybrid includes more sources
