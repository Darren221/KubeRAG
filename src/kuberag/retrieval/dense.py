from typing import Protocol

from kuberag.stores import ChromaStore, Hit


class EmbedderLike(Protocol):
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class DenseRetriever:
    def __init__(self, *, embedder: EmbedderLike, store: ChromaStore) -> None:
        self.embedder = embedder
        self.store = store

    async def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        if not query.strip():
            return []
        embeddings = await self.embedder.embed_batch([query])
        return self.store.query(embeddings[0], k=k)
