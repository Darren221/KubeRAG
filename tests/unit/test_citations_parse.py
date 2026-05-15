import pytest

from kuberag.generation.citations import (
    CitationError,
    ParsedCitation,
    find_text_markers,
    parse_citations,
)
from kuberag.generation.models import Answer, Citation
from kuberag.retrieval.fusion import FusedHit


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    source: str | None = None,
    section: str | None = None,
    rank: int = 0,
) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        text=text,
        source=source or f"/test/{chunk_id}.md",
        section=section,
        chunking_strategy="fixed",
        rrf_score=0.5,
        rank=rank,
        dense_rank=rank,
        sparse_rank=rank,
    )


def make_answer(
    text: str,
    citations: list[tuple[int, str]],
    *,
    insufficient: bool = False,
) -> Answer:
    return Answer(
        text=text,
        citations=[Citation(marker=m, claim_span=s) for m, s in citations],
        insufficient_context=insufficient,
    )


# --- find_text_markers ---


def test_find_text_markers_extracts_numbers() -> None:
    assert find_text_markers("Pods are units [1]. Services route traffic [2].") == {1, 2}


def test_find_text_markers_compound() -> None:
    assert find_text_markers("Multiple support [1][3] this claim.") == {1, 3}


def test_find_text_markers_empty_text() -> None:
    assert find_text_markers("") == set()


def test_find_text_markers_no_brackets() -> None:
    assert find_text_markers("nothing to see here") == set()


def test_find_text_markers_ignores_non_numeric_brackets() -> None:
    assert find_text_markers("Code [block] is not a citation [1].") == {1}


def test_find_text_markers_dedupes() -> None:
    # The same marker repeated in text should appear once in the set
    assert find_text_markers("Pods are units [1]. They run containers [1].") == {1}


# --- parse_citations ---


def test_parses_citations_into_typed_objects() -> None:
    chunks = [make_chunk("a", "pods"), make_chunk("b", "services")]
    answer = make_answer(
        "Pods are units [1]. Services route traffic [2].",
        [(1, "Pods are units"), (2, "Services route traffic")],
    )
    result = parse_citations(answer, chunks)
    assert len(result) == 2
    assert all(isinstance(c, ParsedCitation) for c in result)


def test_parsed_citation_maps_marker_to_chunk() -> None:
    chunks = [make_chunk("a", "pods text"), make_chunk("b", "services text")]
    answer = make_answer(
        "Pods are units [1]. Services [2].",
        [(1, "Pods are units"), (2, "Services")],
    )
    result = parse_citations(answer, chunks)
    assert result[0].chunk_id == "a"
    assert result[0].marker == 1
    assert result[1].chunk_id == "b"
    assert result[1].marker == 2


def test_parsed_citation_includes_chunk_text() -> None:
    chunks = [make_chunk("a", "pod content here")]
    answer = make_answer("Pods [1].", [(1, "Pods")])
    result = parse_citations(answer, chunks)
    assert result[0].chunk_text == "pod content here"


def test_parsed_citation_carries_claim_span() -> None:
    chunks = [make_chunk("a", "pods")]
    answer = make_answer("Pods are units [1].", [(1, "Pods are units")])
    result = parse_citations(answer, chunks)
    assert result[0].claim_span == "Pods are units"


def test_parsed_citation_carries_source_and_section() -> None:
    chunks = [make_chunk("a", "pods", source="/docs/pods.md", section="Pod basics")]
    answer = make_answer("Pods [1].", [(1, "Pods")])
    result = parse_citations(answer, chunks)
    assert result[0].source == "/docs/pods.md"
    assert result[0].section == "Pod basics"


def test_unknown_marker_in_text_raises() -> None:
    chunks = [make_chunk("a", "pods")]
    answer = make_answer(
        "Pods are units [1]. Services [5].",
        [(1, "Pods are units"), (5, "Services")],
    )
    with pytest.raises(CitationError):
        parse_citations(answer, chunks)


def test_unknown_marker_only_in_structured_citations_raises() -> None:
    chunks = [make_chunk("a", "pods")]
    answer = make_answer("Pods are units [1].", [(1, "Pods"), (5, "Out-of-range")])
    with pytest.raises(CitationError):
        parse_citations(answer, chunks)


def test_marker_zero_raises() -> None:
    chunks = [make_chunk("a", "pods")]
    # Pydantic should reject marker=0 (ge=1), but if it leaks through...
    # We can't construct via Citation(marker=0), so test the text-marker path
    answer = Answer(
        text="Pods are units [0].",
        citations=[Citation(marker=1, claim_span="x")],
        insufficient_context=False,
    )
    with pytest.raises(CitationError):
        parse_citations(answer, [chunks[0]])


def test_compound_marker_in_text_produces_two_parsed_citations() -> None:
    chunks = [make_chunk("a", "pods"), make_chunk("b", "services")]
    answer = make_answer(
        "Both passages support this claim [1][2].",
        [(1, "Both passages support this claim"), (2, "Both passages support this claim")],
    )
    result = parse_citations(answer, chunks)
    assert len(result) == 2
    assert {c.chunk_id for c in result} == {"a", "b"}


def test_empty_citations_returns_empty() -> None:
    chunks = [make_chunk("a", "pods")]
    answer = make_answer("No relevant info.", [], insufficient=True)
    assert parse_citations(answer, chunks) == []


def test_insufficient_answer_with_no_text_markers_is_ok() -> None:
    answer = make_answer("Insufficient context to answer.", [], insufficient=True)
    assert parse_citations(answer, []) == []


def test_no_chunks_with_citations_raises() -> None:
    answer = make_answer("Pods [1].", [(1, "Pods")])
    with pytest.raises(CitationError):
        parse_citations(answer, [])


def test_preserves_citation_order() -> None:
    chunks = [make_chunk("a", "pods"), make_chunk("b", "services"), make_chunk("c", "ingress")]
    answer = make_answer(
        "First [2]. Second [1]. Third [3].",
        [(2, "First"), (1, "Second"), (3, "Third")],
    )
    result = parse_citations(answer, chunks)
    assert [c.marker for c in result] == [2, 1, 3]
