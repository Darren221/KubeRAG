from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader

DocumentFormat = Literal["markdown", "html", "text", "pdf"]

_EXTENSION_MAP: dict[str, DocumentFormat] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
    ".pdf": "pdf",
}


class UnsupportedFormatError(ValueError):
    """Raised when a path's extension has no registered loader."""


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: str
    format: DocumentFormat
    text: str
    metadata: dict[str, Any]


def load_document(path: Path) -> Document:
    if not path.exists():
        raise FileNotFoundError(path)

    fmt = _EXTENSION_MAP.get(path.suffix.lower())
    if fmt is None:
        raise UnsupportedFormatError(f"No loader registered for extension '{path.suffix}'")

    text = _LOADERS[fmt](path)
    text = _normalize(text)

    metadata: dict[str, Any] = {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
    }
    if fmt == "pdf":
        metadata["page_count"] = _pdf_page_count(path)

    return Document(source_path=str(path), format=fmt, text=text, metadata=metadata)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    return soup.get_text(separator=" ", strip=True)


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in parts if p.strip())


_LOADERS: dict[DocumentFormat, Any] = {
    "markdown": _load_markdown,
    "html": _load_html,
    "text": _load_text,
    "pdf": _load_pdf,
}


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _pdf_page_count(path: Path) -> int:
    return len(PdfReader(str(path)).pages)
