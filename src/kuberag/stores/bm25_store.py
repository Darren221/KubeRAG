import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from kuberag.ingest.chunkers import Chunk
from kuberag.stores.models import Hit

_TOKEN_RE = re.compile(r"\w[\w-]*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "chunking_strategy": chunk.chunking_strategy,
        "char_count": chunk.char_count,
    }
    if chunk.section is not None:
        meta["section"] = chunk.section
    return meta


class BM25Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chunks: dict[str, Chunk] = {}
        self._tokens: dict[str, list[str]] = {}
        self._bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as f:
            state = pickle.load(f)
        self._chunks = state["chunks"]
        self._tokens = state["tokens"]
        self._rebuild_index()

    def _persist(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"chunks": self._chunks, "tokens": self._tokens}, f)
        tmp.replace(self.path)

    def _rebuild_index(self) -> None:
        ordered = [self._tokens[chunk_id] for chunk_id in self._chunks]
        self._bm25 = BM25Okapi(ordered) if ordered else None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self._tokens[chunk.id] = _tokenize(chunk.text)
        self._rebuild_index()
        self._persist()

    def query(self, text: str, k: int) -> list[Hit]:
        if not self._chunks or self._bm25 is None:
            return []
        query_tokens = _tokenize(text)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        chunk_ids = list(self._chunks.keys())
        ranked = sorted(
            zip(chunk_ids, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )[:k]

        hits: list[Hit] = []
        for rank, (chunk_id, score) in enumerate(ranked):
            chunk = self._chunks[chunk_id]
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    section=chunk.section,
                    chunking_strategy=chunk.chunking_strategy,
                    score=float(score),
                    rank=rank,
                    metadata=_chunk_metadata(chunk),
                )
            )
        return hits

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        """Clear all chunks and delete the persistence file. Destroys all data."""
        self._chunks.clear()
        self._tokens.clear()
        self._bm25 = None
        if self.path.exists():
            self.path.unlink()
