from pathlib import Path

import numpy as np
import pytest

from kuberag.ingest.chunkers import Chunk
from kuberag.stores.chroma_store import ChromaStore, Hit

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


def random_embedding(seed: int, dim: int = 16) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).tolist()


def test_add_and_count(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    chunks = [make_chunk(f"id-{i}", f"text {i}", index=i) for i in range(5)]
    embeddings = [random_embedding(i) for i in range(5)]
    store.add(chunks, embeddings)
    assert store.count() == 5


def test_query_returns_top_match_for_self(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    chunks = [make_chunk(f"id-{i}", f"text {i}", index=i) for i in range(5)]
    embeddings = [random_embedding(i) for i in range(5)]
    store.add(chunks, embeddings)

    hits = store.query(embeddings[2], k=3)
    assert hits[0].chunk_id == "id-2"
    assert hits[0].rank == 0


def test_persistence_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "chroma"

    store1 = ChromaStore(path)
    chunk = make_chunk("persistent", "stays around")
    store1.add([chunk], [random_embedding(0)])

    store2 = ChromaStore(path)
    assert store2.count() == 1


def test_metadata_preserved(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    chunk = make_chunk("id-1", "hello", source="/docs/intro.md", section="Intro")
    store.add([chunk], [random_embedding(0)])

    hits = store.query(random_embedding(0), k=1)
    assert hits[0].source == "/docs/intro.md"
    assert hits[0].section == "Intro"


def test_none_section_round_trips(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    chunk = make_chunk("id-1", "no section", section=None)
    store.add([chunk], [random_embedding(0)])

    hits = store.query(random_embedding(0), k=1)
    assert hits[0].section is None


def test_upsert_does_not_duplicate(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    store.add([make_chunk("dup-id", "v1")], [random_embedding(0)])
    store.add([make_chunk("dup-id", "v2 updated")], [random_embedding(1)])

    assert store.count() == 1
    hits = store.query(random_embedding(1), k=1)
    assert hits[0].text == "v2 updated"


def test_empty_add_is_noop(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    store.add([], [])
    assert store.count() == 0


def test_empty_collection_query_returns_empty(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    assert store.query(random_embedding(0), k=5) == []


def test_query_returns_at_most_k(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    chunks = [make_chunk(f"id-{i}", f"text {i}", index=i) for i in range(10)]
    embeddings = [random_embedding(i) for i in range(10)]
    store.add(chunks, embeddings)

    hits = store.query(random_embedding(0), k=3)
    assert len(hits) == 3


def test_mismatched_lengths_rejected(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    with pytest.raises(ValueError):
        store.add([make_chunk("a", "hi")], [random_embedding(0), random_embedding(1)])


def test_self_match_similarity_near_one(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    emb = [1.0, 0.0, 0.0, 0.0]
    store.add([make_chunk("id-1", "hello")], [emb])

    hits = store.query(emb, k=1)
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_ranks_are_sequential(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    chunks = [make_chunk(f"id-{i}", f"text {i}", index=i) for i in range(5)]
    embeddings = [random_embedding(i) for i in range(5)]
    store.add(chunks, embeddings)

    hits = store.query(random_embedding(0), k=5)
    assert [h.rank for h in hits] == [0, 1, 2, 3, 4]


def test_hit_is_pydantic_model(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    store.add([make_chunk("id-1", "hi")], [random_embedding(0)])
    hits = store.query(random_embedding(0), k=1)
    assert isinstance(hits[0], Hit)
