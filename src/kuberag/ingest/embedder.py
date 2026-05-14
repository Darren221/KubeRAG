import hashlib
from pathlib import Path

import numpy as np
import openai
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def _cache_key(text: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> list[float] | None:
        file = self.path / f"{key}.npy"
        if not file.exists():
            return None
        return np.load(file).tolist()  # type: ignore[no-any-return]

    def put(self, key: str, vector: list[float]) -> None:
        file = self.path / f"{key}.npy"
        np.save(file, np.array(vector, dtype=np.float32))


class Embedder:
    def __init__(
        self,
        *,
        model: str,
        client: AsyncOpenAI,
        cache: EmbeddingCache,
        batch_size: int = 100,
        max_retries: int = 5,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 30.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.client = client
        self.cache = cache
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(_cache_key(text, self.model))
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        for start in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[start : start + self.batch_size]
            batch_indices = uncached_indices[start : start + self.batch_size]
            vectors = await self._embed_with_retry(batch)
            for idx, text, vector in zip(batch_indices, batch, vectors, strict=True):
                results[idx] = vector
                self.cache.put(_cache_key(text, self.model), vector)

        final: list[list[float]] = []
        for r in results:
            assert r is not None
            final.append(r)
        return final

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.retry_min_wait,
                max=max(self.retry_max_wait, self.retry_min_wait),
            ),
            retry=retry_if_exception_type(_RETRYABLE_ERRORS),
            reraise=True,
        ):
            with attempt:
                response = await self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                )
                return [d.embedding for d in response.data]
        raise RuntimeError("retry loop exited without producing a result")
