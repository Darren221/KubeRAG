import hashlib
from pathlib import Path

import numpy as np
import pytest

from kuberag.ingest.chunkers import Chunk
from kuberag.retrieval.dense import DenseRetriever
from kuberag.stores import ChromaStore, Hit

pytestmark = pytest.mark.integration


class FakeEmbedder:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.call_log: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.call_log.append(list(texts))
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        return rng.standard_normal(self.dim).tolist()


def make_chunk(chunk_id: str, text: str, *, index: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source=f"/test/{chunk_id}.md",
        section=None,
        chunk_index=index,
        chunking_strategy="fixed",
        char_count=len(text),
    )


async def populate_store(store: ChromaStore, embedder: FakeEmbedder, texts: list[str]) -> None:
    chunks = [make_chunk(f"c{i}", t, index=i) for i, t in enumerate(texts)]
    embeddings = await embedder.embed_batch(texts)
    store.add(chunks, embeddings)


async def test_retrieves_chunks(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    embedder = FakeEmbedder()
    await populate_store(store, embedder, ["pods are units", "services route traffic", "ingress at the edge"])

    retriever = DenseRetriever(embedder=embedder, store=store)
    hits = await retriever.retrieve("pods are units", k=3)

    assert hits
    assert isinstance(hits[0], Hit)


async def test_embeds_only_the_query(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    embedder = FakeEmbedder()
    await populate_store(store, embedder, ["pods are units"])
    embedder.call_log.clear()

    retriever = DenseRetriever(embedder=embedder, store=store)
    await retriever.retrieve("pods", k=1)

    assert embedder.call_log == [["pods"]]


async def test_top_match_is_the_seeded_chunk(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    embedder = FakeEmbedder()
    await populate_store(
        store,
        embedder,
        [
            "pods are the smallest deployable unit",
            "services expose pods at a stable endpoint",
            "ingress routes external traffic",
        ],
    )

    retriever = DenseRetriever(embedder=embedder, store=store)
    hits = await retriever.retrieve("pods are the smallest deployable unit", k=3)

    assert hits[0].chunk_id == "c0"
    assert hits[0].rank == 0


async def test_respects_k(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    embedder = FakeEmbedder()
    await populate_store(store, embedder, [f"chunk text {i}" for i in range(10)])

    retriever = DenseRetriever(embedder=embedder, store=store)
    hits = await retriever.retrieve("chunk text 0", k=3)
    assert len(hits) == 3


async def test_empty_corpus_returns_empty(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    embedder = FakeEmbedder()
    retriever = DenseRetriever(embedder=embedder, store=store)
    assert await retriever.retrieve("anything", k=5) == []


async def test_hits_preserve_metadata(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    embedder = FakeEmbedder()
    chunk = Chunk(
        id="c0",
        text="pods are units",
        source="/docs/intro.md",
        section="Pods",
        chunk_index=0,
        chunking_strategy="fixed",
        char_count=14,
    )
    [emb] = await embedder.embed_batch(["pods are units"])
    store.add([chunk], [emb])

    retriever = DenseRetriever(embedder=embedder, store=store)
    hits = await retriever.retrieve("pods are units", k=1)

    assert hits[0].source == "/docs/intro.md"
    assert hits[0].section == "Pods"
