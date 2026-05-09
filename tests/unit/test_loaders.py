from pathlib import Path

import pytest

from kuberag.ingest.loaders import Document, UnsupportedFormatError, load_document


def test_loads_markdown(tmp_path: Path) -> None:
    p = tmp_path / "intro.md"
    p.write_text("# Title\n\nSome **bold** text and a [link](https://example.com).\n")
    doc = load_document(p)
    assert isinstance(doc, Document)
    assert doc.format == "markdown"
    assert doc.source_path == str(p)
    assert "**bold**" in doc.text
    assert "Some" in doc.text
    assert doc.metadata["filename"] == "intro.md"


def test_loads_text(tmp_path: Path) -> None:
    p = tmp_path / "notes.txt"
    p.write_text("plain old text\nline two\n")
    doc = load_document(p)
    assert doc.format == "text"
    assert doc.text == "plain old text\nline two\n"
    assert doc.metadata["filename"] == "notes.txt"


def test_loads_html_strips_tags(tmp_path: Path) -> None:
    p = tmp_path / "page.html"
    p.write_text(
        "<html><body>"
        "<h1>Title</h1>"
        "<p>Hello <b>world</b></p>"
        "<script>alert('x');</script>"
        "<style>body { color: red }</style>"
        "</body></html>"
    )
    doc = load_document(p)
    assert doc.format == "html"
    assert "<h1>" not in doc.text
    assert "<script>" not in doc.text
    assert "alert" not in doc.text  # script content stripped
    assert "color: red" not in doc.text  # style content stripped
    assert "Title" in doc.text
    assert "Hello" in doc.text and "world" in doc.text


def test_loads_pdf(pdf_fixture: Path) -> None:
    doc = load_document(pdf_fixture)
    assert doc.format == "pdf"
    assert "Hello PDF" in doc.text
    assert "page two" in doc.text.lower()
    assert doc.metadata["page_count"] == 2


def test_normalizes_line_endings(tmp_path: Path) -> None:
    p = tmp_path / "windows.md"
    p.write_bytes(b"line one\r\nline two\r\n")
    doc = load_document(p)
    assert "\r" not in doc.text
    assert doc.text == "line one\nline two\n"


def test_unsupported_format_raises(tmp_path: Path) -> None:
    p = tmp_path / "weird.xyz"
    p.write_text("???")
    with pytest.raises(UnsupportedFormatError):
        load_document(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "ghost.md"
    with pytest.raises(FileNotFoundError):
        load_document(p)


def test_metadata_includes_size(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello")
    doc = load_document(p)
    assert doc.metadata["size_bytes"] == 5
