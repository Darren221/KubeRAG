import pytest

from kuberag.ingest.chunkers import Chunk, Chunker, RecursiveChunker
from kuberag.ingest.loaders import Document


def make_doc(text: str, path: str = "/test/sample.md") -> Document:
    return Document(source_path=path, format="markdown", text=text, metadata={})


def test_empty_document_produces_no_chunks() -> None:
    chunker = RecursiveChunker(size=400, overlap=50)
    assert chunker.chunk(make_doc("")) == []


def test_chunking_strategy_tag_is_recursive() -> None:
    chunker = RecursiveChunker(size=400, overlap=50)
    chunks = chunker.chunk(make_doc("## Title\n\nSome content here."))
    assert chunks
    assert all(c.chunking_strategy == "recursive" for c in chunks)


def test_doc_without_headings_has_none_section() -> None:
    chunker = RecursiveChunker(size=400, overlap=50)
    text = "Just some plain text without any headings.\n\nAnother paragraph."
    chunks = chunker.chunk(make_doc(text))
    assert chunks
    assert all(c.section is None for c in chunks)


def test_chunks_tagged_with_their_section() -> None:
    chunker = RecursiveChunker(size=400, overlap=50)
    text = (
        "## Pods\nA pod is the smallest deployable unit.\n\n"
        "## Services\nA service exposes pods to the network.\n\n"
        "## Deployments\nA deployment manages replicas of pods.\n"
    )
    chunks = chunker.chunk(make_doc(text))
    sections = {c.section for c in chunks}
    assert {"Pods", "Services", "Deployments"} <= sections


def test_chunk_falls_under_its_own_heading() -> None:
    chunker = RecursiveChunker(size=400, overlap=50)
    text = (
        "## Pods\nPod content describes pods specifically.\n\n"
        "## Services\nService content describes services specifically.\n"
    )
    chunks = chunker.chunk(make_doc(text))
    for c in chunks:
        if "Pod content" in c.text:
            assert c.section == "Pods"
        if "Service content" in c.text:
            assert c.section == "Services"


def test_chunks_respect_size_limit_with_slack() -> None:
    chunker = RecursiveChunker(size=200, overlap=20)
    text = "This is a sentence. " * 50
    chunks = chunker.chunk(make_doc(text))
    # LangChain's recursive splitter may slightly exceed size when a single
    # token is bigger than the limit. Allow generous slack.
    for c in chunks:
        assert c.char_count <= 300


def test_large_section_gets_sub_split() -> None:
    chunker = RecursiveChunker(size=200, overlap=20)
    text = "## Big Section\n\n" + ("Sentence here. " * 30)
    chunks = chunker.chunk(make_doc(text))
    assert len(chunks) > 1
    assert all(c.section == "Big Section" for c in chunks)


def test_h1_h2_h3_all_recognized() -> None:
    chunker = RecursiveChunker(size=400, overlap=50)
    text = "# Top\ntop content\n\n## Middle\nmiddle content\n\n### Inner\ninner content\n"
    chunks = chunker.chunk(make_doc(text))
    sections = {c.section for c in chunks}
    assert {"Top", "Middle", "Inner"} <= sections


def test_hash_without_space_is_not_a_heading() -> None:
    chunker = RecursiveChunker(size=400, overlap=50)
    text = "This line has a #hashtag in the middle.\nAlso #notaheading at the start."
    chunks = chunker.chunk(make_doc(text))
    assert chunks
    assert all(c.section is None for c in chunks)


def test_chunk_indexes_are_sequential() -> None:
    chunker = RecursiveChunker(size=200, overlap=20)
    chunks = chunker.chunk(make_doc("Some text. " * 50))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_deterministic() -> None:
    chunker = RecursiveChunker(size=200, overlap=20)
    text = "## Title\n\nSome text here.\n\nMore text follows."
    a = chunker.chunk(make_doc(text))
    b = chunker.chunk(make_doc(text))
    assert [c.id for c in a] == [c.id for c in b]


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(size=100, overlap=100)
    with pytest.raises(ValueError):
        RecursiveChunker(size=100, overlap=200)


def test_satisfies_chunker_protocol() -> None:
    chunker: Chunker = RecursiveChunker(size=400, overlap=50)
    assert chunker.name == "recursive"


def test_source_propagates_from_document() -> None:
    chunker = RecursiveChunker(size=400, overlap=50)
    chunks = chunker.chunk(make_doc("hello", path="/docs/foo.md"))
    assert all(c.source == "/docs/foo.md" for c in chunks)


def test_chunks_are_pydantic_models() -> None:
    chunker = RecursiveChunker(size=400, overlap=50)
    chunks = chunker.chunk(make_doc("hello"))
    assert all(isinstance(c, Chunk) for c in chunks)


def test_empty_section_body_skipped() -> None:
    chunker = RecursiveChunker(size=400, overlap=50)
    # Header with no body should not produce a chunk
    text = "## Pods\n## Services\nactual content\n"
    chunks = chunker.chunk(make_doc(text))
    assert all(c.text.strip() for c in chunks)
    assert any(c.section == "Services" for c in chunks)
