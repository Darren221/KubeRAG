import hashlib
from typing import Protocol

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
