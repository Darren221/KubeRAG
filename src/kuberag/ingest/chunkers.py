import hashlib
import re
from typing import Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field

from kuberag.ingest.loaders import Document


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    source: str
    section: str | None = None
    chunk_index: int = Field(ge=0)
    chunking_strategy: str
    char_count: int = Field(ge=0)


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def _chunk_id(source: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(str(chunk_index).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _split_by_headings(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.splitlines(keepends=True):
        match = _HEADING_RE.match(line)
        if match:
            heading_text = match.group(2).strip()
            sections.append((heading_text, []))
        else:
            sections[-1][1].append(line)
    return [(name, "".join(lines)) for name, lines in sections]


class FixedSizeChunker:
    name = "fixed"

    def __init__(self, *, size: int = 800, overlap: int = 120) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []

        stride = self.size - self.overlap
        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.size, len(text))
            piece = text[start:end]
            chunks.append(
                Chunk(
                    id=_chunk_id(doc.source_path, index, piece),
                    text=piece,
                    source=doc.source_path,
                    section=None,
                    chunk_index=index,
                    chunking_strategy=self.name,
                    char_count=len(piece),
                )
            )
            if end == len(text):
                break
            start += stride
            index += 1

        return chunks


class RecursiveChunker:
    name = "recursive"

    def __init__(self, *, size: int = 800, overlap: int = 120) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False,
            length_function=len,
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text:
            return []

        sections = _split_by_headings(doc.text)
        chunks: list[Chunk] = []
        index = 0

        for section_name, body in sections:
            if not body.strip():
                continue
            for piece in self._splitter.split_text(body):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        id=_chunk_id(doc.source_path, index, piece),
                        text=piece,
                        source=doc.source_path,
                        section=section_name,
                        chunk_index=index,
                        chunking_strategy=self.name,
                        char_count=len(piece),
                    )
                )
                index += 1

        return chunks
