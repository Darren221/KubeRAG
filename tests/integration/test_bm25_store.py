from pathlib import Path

import pytest

from kuberag.ingest.chunkers import Chunk
from kuberag.stores import BM25Store, Hit

pytestmark = pytest.mark.integration


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    source: str = "/test/doc.md",
    section: str | None = None,
    index: int = 0,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source=source,
        section=section,
        chunk_index=index,
        chunking_strategy="fixed",
        char_count=len(text),
    )


def test_empty_store_count_is_zero(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    assert store.count() == 0


def test_add_and_count(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add([make_chunk(f"id-{i}", f"text {i}", index=i) for i in range(5)])
    assert store.count() == 5


def test_query_finds_rare_token(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add(
        [
            make_chunk("a", "Some text about pods running in the cluster.", index=0),
            make_chunk("b", "Services expose pods to the network.", index=1),
            make_chunk("c", "The kubelet-hostpath driver mounts host paths.", index=2),
            make_chunk("d", "Deployments manage pod replicas.", index=3),
            make_chunk("e", "Volumes provide persistent storage.", index=4),
        ]
    )
    hits = store.query("kubelet-hostpath", k=3)
    assert hits
    assert hits[0].chunk_id == "c"


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "bm25.pkl"
    store1 = BM25Store(path)
    store1.add([make_chunk("persistent", "stays around forever", index=0)])

    store2 = BM25Store(path)
    assert store2.count() == 1
    hits = store2.query("stays", k=1)
    assert hits[0].chunk_id == "persistent"


def test_empty_add_is_noop(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add([])
    assert store.count() == 0


def test_empty_corpus_query_returns_empty(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    assert store.query("anything", k=5) == []


def test_empty_query_text_returns_empty(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add([make_chunk("a", "hello world", index=0)])
    assert store.query("", k=5) == []


def test_query_returns_at_most_k(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add([make_chunk(f"id-{i}", f"term-{i} sample text", index=i) for i in range(10)])
    hits = store.query("term-3 term-1 term-9", k=3)
    assert len(hits) <= 3


def test_upsert_replaces_by_id(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add([make_chunk("a", "first version of the chunk", index=0)])
    store.add([make_chunk("a", "updated version of the chunk", index=0)])
    assert store.count() == 1
    hits = store.query("updated", k=1)
    assert hits[0].chunk_id == "a"
    assert "updated" in hits[0].text


def test_ranks_are_sequential(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add(
        [
            make_chunk(f"id-{i}", f"term-{i} appears here exactly once", index=i)
            for i in range(5)
        ]
    )
    hits = store.query("term-2 term-1 term-0", k=5)
    assert [h.rank for h in hits] == list(range(len(hits)))


def test_metadata_in_hit(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add(
        [make_chunk("a", "hello there world", source="/x.md", section="Intro", index=0)]
    )
    hits = store.query("hello", k=1)
    assert hits[0].source == "/x.md"
    assert hits[0].section == "Intro"
    assert hits[0].chunking_strategy == "fixed"


def test_hit_is_pydantic_model(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add([make_chunk("a", "hello world", index=0)])
    hits = store.query("hello", k=1)
    assert isinstance(hits[0], Hit)


def test_case_insensitive_match(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "bm25.pkl")
    store.add([make_chunk("a", "Kubelet is responsible for node-level operations.", index=0)])
    hits = store.query("kubelet", k=1)
    assert hits and hits[0].chunk_id == "a"
