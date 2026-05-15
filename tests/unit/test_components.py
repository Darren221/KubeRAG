from dashboard.components import linkify_citations


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
