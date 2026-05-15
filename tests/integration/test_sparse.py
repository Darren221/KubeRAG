from pathlib import Path

import pytest

from kuberag.ingest.chunkers import Chunk
from kuberag.retrieval.sparse import SparseRetriever
from kuberag.stores import BM25Store, Hit

pytestmark = pytest.mark.integration


def make_chunk(chunk_id: str, text: str, *, index: int = 0, section: str | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source=f"/test/{chunk_id}.md",
        section=section,
        chunk_index=index,
        chunking_strategy="fixed",
        char_count=len(text),
    )


async def test_retrieves_keyword_match(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add(
        [
            make_chunk("a", "pods are units of work", index=0),
            make_chunk("b", "services route traffic", index=1),
            make_chunk("c", "ingress sits at the edge", index=2),
        ]
    )
    retriever = SparseRetriever(store=store)
    hits = await retriever.retrieve("pods", k=3)
    assert hits
    assert hits[0].chunk_id == "a"
    assert isinstance(hits[0], Hit)


async def test_rare_token_returns_top_one(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add(
        [
            make_chunk("a", "pods run containers in the cluster", index=0),
            make_chunk("b", "services route traffic to pods", index=1),
            make_chunk("c", "the kubelet-hostpath driver mounts host paths", index=2),
            make_chunk("d", "deployments manage pod replicas", index=3),
        ]
    )
    retriever = SparseRetriever(store=store)
    hits = await retriever.retrieve("kubelet-hostpath", k=3)
    assert hits[0].chunk_id == "c"


async def test_empty_query_returns_empty(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add([make_chunk("a", "hello world", index=0)])
    retriever = SparseRetriever(store=store)
    assert await retriever.retrieve("", k=3) == []
    assert await retriever.retrieve("   ", k=3) == []


async def test_empty_corpus_returns_empty(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    retriever = SparseRetriever(store=store)
    assert await retriever.retrieve("anything", k=5) == []


async def test_respects_k(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add([make_chunk(f"id-{i}", f"term-{i} appears here", index=i) for i in range(10)])
    retriever = SparseRetriever(store=store)
    hits = await retriever.retrieve("term-3 term-1 term-9", k=2)
    assert len(hits) <= 2


async def test_metadata_preserved(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add([make_chunk("a", "hello there world", section="Intro", index=0)])
    retriever = SparseRetriever(store=store)
    hits = await retriever.retrieve("hello", k=1)
    assert hits[0].source == "/test/a.md"
    assert hits[0].section == "Intro"
    assert hits[0].chunking_strategy == "fixed"


async def test_ranks_are_sequential(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add(
        [make_chunk(f"id-{i}", f"term-{i} sample content", index=i) for i in range(5)]
    )
    retriever = SparseRetriever(store=store)
    hits = await retriever.retrieve("term-0 term-1 term-2", k=5)
    assert [h.rank for h in hits] == list(range(len(hits)))
