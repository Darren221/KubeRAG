from pathlib import Path

import chromadb
import numpy as np

from kuberag.ingest.chunkers import Chunk
from kuberag.stores.models import DocumentSummary, Hit

_DEFAULT_COLLECTION = "kuberag"


def _chunk_to_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    meta: dict[str, str | int | float | bool] = {
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "chunking_strategy": chunk.chunking_strategy,
        "char_count": chunk.char_count,
    }
    if chunk.section is not None:
        meta["section"] = chunk.section
    return meta


class ChromaStore:
    def __init__(self, path: Path, collection_name: str = _DEFAULT_COLLECTION) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop the collection and recreate it empty. Destroys all data."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=np.asarray(embeddings, dtype=np.float32),
            documents=[c.text for c in chunks],
            metadatas=[_chunk_to_metadata(c) for c in chunks],
        )

    def query(self, query_embedding: list[float], k: int) -> list[Hit]:
        total = self.count()
        if total == 0:
            return []

        result = self._collection.query(
            query_embeddings=np.asarray([query_embedding], dtype=np.float32),
            n_results=min(k, total),
        )
        assert result["ids"] is not None
        assert result["documents"] is not None
        assert result["distances"] is not None
        assert result["metadatas"] is not None
        ids = result["ids"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]

        hits: list[Hit] = []
        for rank, (chunk_id, text, distance, metadata) in enumerate(
            zip(ids, documents, distances, metadatas, strict=True)
        ):
            meta = dict(metadata) if metadata else {}
            section = meta.get("section")
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    text=text,
                    source=str(meta.get("source", "")),
                    section=str(section) if section else None,
                    chunking_strategy=str(meta.get("chunking_strategy", "")),
                    score=1.0 - float(distance),
                    rank=rank,
                    metadata=meta,
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def all_embeddings(self) -> list[list[float]]:
        if self.count() == 0:
            return []
        result = self._collection.get(include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None:
            return []
        return [list(map(float, vec)) for vec in embeddings]

    def list_documents(self) -> list[DocumentSummary]:
        if self.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        counts: dict[tuple[str, str], int] = {}
        for meta in metadatas:
            if not meta:
                continue
            source = str(meta.get("source", ""))
            strategy = str(meta.get("chunking_strategy", ""))
            counts[(source, strategy)] = counts.get((source, strategy), 0) + 1

        return [
            DocumentSummary(
                source=source, chunking_strategy=strategy, chunk_count=count
            )
            for (source, strategy), count in sorted(counts.items())
        ]
