from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from kuberag.ingest.embedder import Embedder, EmbeddingCache


def _fake_response(embeddings: list[list[float]]) -> Any:
    response = MagicMock()
    response.data = [MagicMock(embedding=e) for e in embeddings]
    return response


def _rate_limit_error() -> openai.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    return openai.RateLimitError("rate limit", response=response, body=None)


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "embedding_cache"


@pytest.fixture
def cache(cache_path: Path) -> EmbeddingCache:
    return EmbeddingCache(cache_path)


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.embeddings = AsyncMock()
    return client


@pytest.fixture
def embedder(mock_client: AsyncMock, cache: EmbeddingCache) -> Embedder:
    return Embedder(
        model="text-embedding-3-small",
        client=mock_client,
        cache=cache,
        batch_size=100,
        retry_min_wait=0.0,
        retry_max_wait=0.0,
        max_retries=3,
    )


async def test_embeds_single_text(embedder: Embedder, mock_client: AsyncMock) -> None:
    mock_client.embeddings.create.return_value = _fake_response([[0.1, 0.2, 0.3]])
    result = await embedder.embed_batch(["hello"])
    assert result == [[0.1, 0.2, 0.3]]
    mock_client.embeddings.create.assert_called_once_with(
        input=["hello"], model="text-embedding-3-small"
    )


async def test_empty_input_makes_no_api_call(embedder: Embedder, mock_client: AsyncMock) -> None:
    result = await embedder.embed_batch([])
    assert result == []
    mock_client.embeddings.create.assert_not_called()


async def test_batches_large_input(embedder: Embedder, mock_client: AsyncMock) -> None:
    texts = [f"text-{i}" for i in range(250)]
    mock_client.embeddings.create.side_effect = [
        _fake_response([[float(i)] for i in range(100)]),
        _fake_response([[float(i)] for i in range(100, 200)]),
        _fake_response([[float(i)] for i in range(200, 250)]),
    ]
    result = await embedder.embed_batch(texts)
    assert len(result) == 250
    assert mock_client.embeddings.create.call_count == 3


async def test_cache_hit_skips_api(embedder: Embedder, mock_client: AsyncMock) -> None:
    mock_client.embeddings.create.return_value = _fake_response([[0.1, 0.2, 0.3]])
    await embedder.embed_batch(["hello"])
    result = await embedder.embed_batch(["hello"])
    assert mock_client.embeddings.create.call_count == 1
    assert result[0] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)


async def test_mixed_only_uncached_hits_api(
    embedder: Embedder, mock_client: AsyncMock
) -> None:
    mock_client.embeddings.create.return_value = _fake_response([[0.9, 0.9, 0.9]])
    await embedder.embed_batch(["hello"])
    mock_client.embeddings.create.reset_mock()

    mock_client.embeddings.create.return_value = _fake_response([[0.1, 0.1, 0.1]])
    result = await embedder.embed_batch(["hello", "world"])

    mock_client.embeddings.create.assert_called_once_with(
        input=["world"], model="text-embedding-3-small"
    )
    assert result[0] == pytest.approx([0.9, 0.9, 0.9], abs=1e-6)
    assert result[1] == pytest.approx([0.1, 0.1, 0.1], abs=1e-6)


async def test_order_preserved_with_mixed_cache(
    embedder: Embedder, mock_client: AsyncMock
) -> None:
    mock_client.embeddings.create.return_value = _fake_response([[0.2]])
    await embedder.embed_batch(["b"])
    mock_client.embeddings.create.reset_mock()

    mock_client.embeddings.create.return_value = _fake_response([[0.1], [0.3]])
    result = await embedder.embed_batch(["a", "b", "c"])
    assert result[0] == pytest.approx([0.1], abs=1e-6)
    assert result[1] == pytest.approx([0.2], abs=1e-6)
    assert result[2] == pytest.approx([0.3], abs=1e-6)


async def test_retries_on_rate_limit(embedder: Embedder, mock_client: AsyncMock) -> None:
    mock_client.embeddings.create.side_effect = [
        _rate_limit_error(),
        _fake_response([[0.5, 0.5, 0.5]]),
    ]
    result = await embedder.embed_batch(["hello"])
    assert result == [[0.5, 0.5, 0.5]]
    assert mock_client.embeddings.create.call_count == 2


async def test_gives_up_after_max_retries(
    embedder: Embedder, mock_client: AsyncMock
) -> None:
    mock_client.embeddings.create.side_effect = _rate_limit_error()
    with pytest.raises(openai.RateLimitError):
        await embedder.embed_batch(["hello"])
    assert mock_client.embeddings.create.call_count == 3


async def test_cache_persists_to_disk(
    cache_path: Path, mock_client: AsyncMock
) -> None:
    e1 = Embedder(
        model="text-embedding-3-small",
        client=mock_client,
        cache=EmbeddingCache(cache_path),
        batch_size=100,
        retry_min_wait=0.0,
        retry_max_wait=0.0,
    )
    mock_client.embeddings.create.return_value = _fake_response([[0.1, 0.2, 0.3]])
    await e1.embed_batch(["hello"])
    mock_client.embeddings.create.reset_mock()

    e2 = Embedder(
        model="text-embedding-3-small",
        client=mock_client,
        cache=EmbeddingCache(cache_path),
        batch_size=100,
        retry_min_wait=0.0,
        retry_max_wait=0.0,
    )
    result = await e2.embed_batch(["hello"])
    mock_client.embeddings.create.assert_not_called()
    assert result[0] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)


async def test_different_models_have_different_cache_keys(
    cache_path: Path, mock_client: AsyncMock
) -> None:
    cache = EmbeddingCache(cache_path)
    small = Embedder(
        model="text-embedding-3-small",
        client=mock_client,
        cache=cache,
        retry_min_wait=0.0,
        retry_max_wait=0.0,
    )
    large = Embedder(
        model="text-embedding-3-large",
        client=mock_client,
        cache=cache,
        retry_min_wait=0.0,
        retry_max_wait=0.0,
    )

    mock_client.embeddings.create.return_value = _fake_response([[0.1]])
    await small.embed_batch(["hello"])

    mock_client.embeddings.create.reset_mock()
    mock_client.embeddings.create.return_value = _fake_response([[0.2]])
    await large.embed_batch(["hello"])

    mock_client.embeddings.create.assert_called_once()


def test_batch_size_must_be_positive(mock_client: AsyncMock, cache: EmbeddingCache) -> None:
    with pytest.raises(ValueError):
        Embedder(model="m", client=mock_client, cache=cache, batch_size=0)
    with pytest.raises(ValueError):
        Embedder(model="m", client=mock_client, cache=cache, batch_size=-1)
