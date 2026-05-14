import logging
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from kuberag.ingest.chunkers import Chunk, Chunker
from kuberag.ingest.dedupe import nearest_similarity
from kuberag.ingest.loaders import load_document
from kuberag.stores import BM25Store, ChromaStore

logger = logging.getLogger(__name__)


class EmbedderLike(Protocol):
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class SkippedChunk(BaseModel):
    chunk_id: str
    similarity: float
    nearest_chunk_id: str | None = None


class IngestResult(BaseModel):
    docs_loaded: int
    chunks_produced: int
    chunks_inserted: int
    chunks_skipped_duplicate: int
    chroma_count: int
    bm25_count: int
    skipped: list[SkippedChunk] = Field(default_factory=list)


class IngestIntegrityError(RuntimeError):
    pass


class IngestPipeline:
    def __init__(
        self,
        *,
        chunkers: dict[str, Chunker],
        embedder: EmbedderLike,
        chroma_store: ChromaStore,
        bm25_store: BM25Store,
        dedupe_threshold: float = 0.95,
    ) -> None:
        if not chunkers:
            raise ValueError("chunkers must contain at least one strategy")
        self.chunkers = chunkers
        self.embedder = embedder
        self.chroma_store = chroma_store
        self.bm25_store = bm25_store
        self.dedupe_threshold = dedupe_threshold

    async def run(
        self,
        paths: list[Path],
        *,
        chunker_name: str = "fixed",
    ) -> IngestResult:
        if chunker_name not in self.chunkers:
            raise ValueError(
                f"Unknown chunker '{chunker_name}'. Available: {sorted(self.chunkers)}"
            )
        chunker = self.chunkers[chunker_name]

        all_chunks: list[Chunk] = []
        for path in paths:
            doc = load_document(path)
            all_chunks.extend(chunker.chunk(doc))

        if not all_chunks:
            return self._empty_result(len(paths))

        embeddings = await self.embedder.embed_batch([c.text for c in all_chunks])

        existing = self.chroma_store.all_embeddings()
        kept_chunks: list[Chunk] = []
        kept_embeddings: list[list[float]] = []
        skipped: list[SkippedChunk] = []

        for chunk, emb in zip(all_chunks, embeddings, strict=True):
            candidates = existing + kept_embeddings
            similarity, _ = nearest_similarity(emb, candidates)
            if similarity >= self.dedupe_threshold:
                logger.info(
                    "skipping near-duplicate chunk %s (similarity=%.3f)",
                    chunk.id,
                    similarity,
                )
                skipped.append(
                    SkippedChunk(chunk_id=chunk.id, similarity=float(similarity))
                )
            else:
                kept_chunks.append(chunk)
                kept_embeddings.append(emb)

        if kept_chunks:
            self.chroma_store.add(kept_chunks, kept_embeddings)
            self.bm25_store.add(kept_chunks)

        chroma_count = self.chroma_store.count()
        bm25_count = self.bm25_store.count()
        if chroma_count != bm25_count:
            raise IngestIntegrityError(
                f"chroma.count()={chroma_count} != bm25.count()={bm25_count}"
            )

        return IngestResult(
            docs_loaded=len(paths),
            chunks_produced=len(all_chunks),
            chunks_inserted=len(kept_chunks),
            chunks_skipped_duplicate=len(skipped),
            chroma_count=chroma_count,
            bm25_count=bm25_count,
            skipped=skipped,
        )

    def _empty_result(self, docs_loaded: int) -> IngestResult:
        return IngestResult(
            docs_loaded=docs_loaded,
            chunks_produced=0,
            chunks_inserted=0,
            chunks_skipped_duplicate=0,
            chroma_count=self.chroma_store.count(),
            bm25_count=self.bm25_store.count(),
        )
