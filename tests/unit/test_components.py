from dashboard.components import format_chunk_provenance, linkify_citations

from kuberag.retrieval.fusion import FusedHit


def _make_hit(
    *,
    rank: int = 0,
    rrf_score: float = 0.0164,
    dense_rank: int | None = 0,
    sparse_rank: int | None = 0,
) -> FusedHit:
    return FusedHit(
        chunk_id="c",
        text="text",
        source="/test/c.md",
        section=None,
        chunking_strategy="fixed",
        rrf_score=rrf_score,
        rank=rank,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
    )


def test_no_markers_returns_unchanged() -> None:
    assert linkify_citations("plain text with no citations") == "plain text with no citations"


def test_single_marker_replaced() -> None:
    out = linkify_citations("Pods are units [1].")
    assert 'href="#chunk-1"' in out
    assert ">[1]<" in out


def test_multiple_markers_replaced() -> None:
    out = linkify_citations("First [1]. Second [2]. Third [3].")
    for n in (1, 2, 3):
        assert f'href="#chunk-{n}"' in out


def test_compound_markers_each_replaced() -> None:
    out = linkify_citations("Both [1][3] support this.")
    assert 'href="#chunk-1"' in out
    assert 'href="#chunk-3"' in out


def test_non_citation_brackets_untouched() -> None:
    # Markdown code blocks have brackets but not citation form
    out = linkify_citations("Use `kubectl logs [pod-name]` to fetch logs.")
    assert "[pod-name]" in out


def test_two_digit_markers_supported() -> None:
    out = linkify_citations("Way over here [10] and over there [12].")
    assert 'href="#chunk-10"' in out
    assert 'href="#chunk-12"' in out


def test_preserves_surrounding_text() -> None:
    out = linkify_citations("Before [1] after.")
    assert "Before " in out
    assert " after." in out


def test_output_uses_css_class() -> None:
    # So we can style citation badges
    out = linkify_citations("Cite [1].")
    assert "kr-citation" in out


def test_provenance_includes_rank_and_rrf_score() -> None:
    line = format_chunk_provenance(_make_hit(rank=2, rrf_score=0.0123))
    assert "rank 2" in line
    assert "0.0123" in line


def test_provenance_includes_both_source_ranks_when_present() -> None:
    line = format_chunk_provenance(_make_hit(dense_rank=1, sparse_rank=3))
    assert "dense #1" in line
    assert "sparse #3" in line


def test_provenance_omits_sparse_when_dense_only() -> None:
    line = format_chunk_provenance(_make_hit(dense_rank=0, sparse_rank=None))
    assert "dense #0" in line
    assert "sparse" not in line


def test_provenance_omits_dense_when_sparse_only() -> None:
    line = format_chunk_provenance(_make_hit(dense_rank=None, sparse_rank=4))
    assert "sparse #4" in line
    assert "dense" not in line


def test_provenance_uses_dot_separator() -> None:
    line = format_chunk_provenance(_make_hit())
    assert " · " in line


def test_linkify_with_custom_anchor_prefix() -> None:
    out = linkify_citations("Cite [1] and [2].", anchor_prefix="hybrid-chunk-")
    assert 'href="#hybrid-chunk-1"' in out
    assert 'href="#hybrid-chunk-2"' in out
    assert "#chunk-1" not in out


def test_default_anchor_prefix_is_chunk_dash() -> None:
    out = linkify_citations("Cite [1].")
    assert 'href="#chunk-1"' in out
