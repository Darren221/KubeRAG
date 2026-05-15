from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kuberag.retrieval.fusion import FusedHit
from kuberag.retrieval.reranker import RerankedIndex, Reranker, RerankResponse

pytestmark = pytest.mark.integration


def make_fused_hit(chunk_id: str, *, rank: int = 0, text: str | None = None) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        text=text or f"text for {chunk_id}",
        source=f"/test/{chunk_id}.md",
        section=None,
        chunking_strategy="fixed",
        rrf_score=0.5,
        rank=rank,
        dense_rank=rank,
        sparse_rank=rank,
    )


def fake_parse_response(ranked_indexes: list[int]) -> Any:
    parsed = RerankResponse(
        ranked=[
            RerankedIndex(index=i, reason="mock reason") for i in ranked_indexes
        ]
    )
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(parsed=parsed))]
    return response


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.beta.chat.completions.parse = AsyncMock()
    return client


@pytest.fixture
def reranker(mock_client: AsyncMock) -> Reranker:
    return Reranker(client=mock_client, model="gpt-4o-mini", top_n=5)


async def test_returns_top_n_in_llm_order(reranker: Reranker, mock_client: AsyncMock) -> None:
    candidates = [make_fused_hit(c, rank=i) for i, c in enumerate(["a", "b", "c", "d"])]
    # LLM says order is c, a, d, b
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([2, 0, 3, 1])
    result = await reranker.rerank("any question", candidates)
    assert [h.chunk_id for h in result] == ["c", "a", "d", "b"]


async def test_returns_at_most_top_n(reranker: Reranker, mock_client: AsyncMock) -> None:
    candidates = [make_fused_hit(c, rank=i) for i, c in enumerate(["a", "b", "c", "d", "e", "f", "g"])]
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([0, 1, 2, 3, 4, 5, 6])
    result = await reranker.rerank("any question", candidates)
    assert len(result) == 5


async def test_rank_field_renumbered_zero_to_top_n_minus_1(
    reranker: Reranker, mock_client: AsyncMock
) -> None:
    candidates = [make_fused_hit(c, rank=99) for c in ["a", "b", "c"]]
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([2, 0, 1])
    result = await reranker.rerank("any question", candidates)
    assert [h.rank for h in result] == [0, 1, 2]


async def test_empty_candidates_makes_no_api_call(
    reranker: Reranker, mock_client: AsyncMock
) -> None:
    result = await reranker.rerank("any question", [])
    assert result == []
    mock_client.beta.chat.completions.parse.assert_not_called()


async def test_single_call_to_llm_regardless_of_candidate_count(
    reranker: Reranker, mock_client: AsyncMock
) -> None:
    candidates = [make_fused_hit(c, rank=i) for i, c in enumerate(["a", "b", "c", "d", "e"])]
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([0, 1, 2, 3, 4])
    await reranker.rerank("any question", candidates)
    assert mock_client.beta.chat.completions.parse.call_count == 1


async def test_shuffled_input_returns_gold_at_top(
    reranker: Reranker, mock_client: AsyncMock
) -> None:
    candidates = [
        make_fused_hit("noise-1", rank=0),
        make_fused_hit("gold", rank=1, text="this answers the question"),
        make_fused_hit("noise-2", rank=2),
    ]
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([1, 0, 2])
    result = await reranker.rerank("the question", candidates)
    assert result[0].chunk_id == "gold"


async def test_excluded_indexes_drop_from_output(
    reranker: Reranker, mock_client: AsyncMock
) -> None:
    # LLM judges that only "a" is relevant; excludes others
    candidates = [make_fused_hit(c, rank=i) for i, c in enumerate(["a", "b", "c"])]
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([0])
    result = await reranker.rerank("any question", candidates)
    assert [h.chunk_id for h in result] == ["a"]


async def test_invalid_indexes_are_ignored(
    reranker: Reranker, mock_client: AsyncMock
) -> None:
    candidates = [make_fused_hit(c, rank=i) for i, c in enumerate(["a", "b"])]
    # LLM hallucinates an index out of range
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([1, 5, 0])
    result = await reranker.rerank("any question", candidates)
    assert [h.chunk_id for h in result] == ["b", "a"]


async def test_duplicate_indexes_are_deduplicated(
    reranker: Reranker, mock_client: AsyncMock
) -> None:
    candidates = [make_fused_hit(c, rank=i) for i, c in enumerate(["a", "b", "c"])]
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([0, 0, 1])
    result = await reranker.rerank("any question", candidates)
    assert [h.chunk_id for h in result] == ["a", "b"]


async def test_prompt_contains_candidates_and_question(
    reranker: Reranker, mock_client: AsyncMock
) -> None:
    candidates = [make_fused_hit("a", rank=0, text="pods are units")]
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([0])
    await reranker.rerank("what is a pod?", candidates)

    call = mock_client.beta.chat.completions.parse.call_args
    messages = call.kwargs["messages"]
    user_message = next(m for m in messages if m["role"] == "user")
    assert "what is a pod?" in user_message["content"]
    assert "pods are units" in user_message["content"]


async def test_uses_configured_model(mock_client: AsyncMock) -> None:
    reranker = Reranker(client=mock_client, model="custom-model", top_n=3)
    candidates = [make_fused_hit("a", rank=0)]
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([0])
    await reranker.rerank("any question", candidates)
    call = mock_client.beta.chat.completions.parse.call_args
    assert call.kwargs["model"] == "custom-model"


async def test_response_format_is_pydantic_schema(
    reranker: Reranker, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response([0])
    await reranker.rerank("any question", [make_fused_hit("a", rank=0)])
    call = mock_client.beta.chat.completions.parse.call_args
    assert call.kwargs["response_format"] is RerankResponse


def test_top_n_must_be_positive(mock_client: AsyncMock) -> None:
    with pytest.raises(ValueError):
        Reranker(client=mock_client, model="m", top_n=0)
    with pytest.raises(ValueError):
        Reranker(client=mock_client, model="m", top_n=-1)
