import pytest

from kuberag.ingest.chunkers import Chunk, Chunker, FixedSizeChunker
from kuberag.ingest.loaders import Document


def make_doc(text: str, path: str = "/test/sample.md") -> Document:
    return Document(source_path=path, format="markdown", text=text, metadata={})


def test_empty_document_produces_no_chunks() -> None:
    chunker = FixedSizeChunker(size=100, overlap=20)
    assert chunker.chunk(make_doc("")) == []


def test_short_document_produces_single_chunk() -> None:
    chunker = FixedSizeChunker(size=800, overlap=120)
    chunks = chunker.chunk(make_doc("short text"))
    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunking_strategy == "fixed"


def test_chunks_overlap_correctly() -> None:
    text = "a" * 200
    chunker = FixedSizeChunker(size=100, overlap=20)
    chunks = chunker.chunk(make_doc(text))
    assert [c.char_count for c in chunks] == [100, 100, 40]
    assert chunks[0].text[-20:] == chunks[1].text[:20]
    assert chunks[1].text[-20:] == chunks[2].text[:20]


def test_no_overlap_when_overlap_zero() -> None:
    chunker = FixedSizeChunker(size=4, overlap=0)
    chunks = chunker.chunk(make_doc("abcdefghij"))
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]


def test_chunk_indexes_are_sequential() -> None:
    chunker = FixedSizeChunker(size=50, overlap=10)
    chunks = chunker.chunk(make_doc("x" * 200))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic() -> None:
    chunker = FixedSizeChunker(size=50, overlap=10)
    doc = make_doc("hello " * 20)
    a = chunker.chunk(doc)
    b = chunker.chunk(doc)
    assert [c.id for c in a] == [c.id for c in b]


def test_chunk_ids_are_unique_within_doc() -> None:
    chunker = FixedSizeChunker(size=50, overlap=10)
    chunks = chunker.chunk(make_doc("hello " * 20))
    assert len({c.id for c in chunks}) == len(chunks)


def test_char_count_matches_text() -> None:
    chunker = FixedSizeChunker(size=100, overlap=20)
    chunks = chunker.chunk(make_doc("a" * 300))
    assert all(c.char_count == len(c.text) for c in chunks)


def test_source_propagates_from_document() -> None:
    chunker = FixedSizeChunker(size=100, overlap=20)
    chunks = chunker.chunk(make_doc("hello", path="/docs/foo.md"))
    assert all(c.source == "/docs/foo.md" for c in chunks)


def test_chunking_strategy_tag_is_fixed() -> None:
    chunker = FixedSizeChunker(size=100, overlap=20)
    chunks = chunker.chunk(make_doc("hello world " * 20))
    assert all(c.chunking_strategy == "fixed" for c in chunks)


def test_section_is_none_for_fixed_chunker() -> None:
    chunker = FixedSizeChunker(size=100, overlap=20)
    chunks = chunker.chunk(make_doc("hello world " * 20))
    assert all(c.section is None for c in chunks)


def test_chunk_count_matches_formula() -> None:
    chunker = FixedSizeChunker(size=100, overlap=20)
    # L=1000, stride=80 → starts at 0, 80, ..., 960; final chunk is partial.
    # ceil((L - overlap) / (size - overlap)) = ceil(980/80) = 13
    chunks = chunker.chunk(make_doc("a" * 1000))
    assert len(chunks) == 13


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(size=100, overlap=100)
    with pytest.raises(ValueError):
        FixedSizeChunker(size=100, overlap=200)


def test_negative_size_or_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(size=0, overlap=0)
    with pytest.raises(ValueError):
        FixedSizeChunker(size=100, overlap=-1)


def test_satisfies_chunker_protocol() -> None:
    chunker: Chunker = FixedSizeChunker(size=100, overlap=20)
    assert chunker.name == "fixed"


def test_chunks_are_immutable() -> None:
    chunker = FixedSizeChunker(size=100, overlap=20)
    chunks = chunker.chunk(make_doc("hello world"))
    with pytest.raises(Exception):
        chunks[0].text = "mutated"  # type: ignore[misc]


def test_chunk_is_pydantic_model() -> None:
    chunker = FixedSizeChunker(size=100, overlap=20)
    chunks = chunker.chunk(make_doc("hello"))
    assert isinstance(chunks[0], Chunk)
