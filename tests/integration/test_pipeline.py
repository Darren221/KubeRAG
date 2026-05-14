import hashlib
from pathlib import Path

import numpy as np
import pytest

from kuberag.ingest.chunkers import FixedSizeChunker, RecursiveChunker
from kuberag.ingest.pipeline import (
    IngestIntegrityError,
    IngestPipeline,
    IngestResult,
)
from kuberag.stores import BM25Store, ChromaStore

pytestmark = pytest.mark.integration

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "docs"


class FakeEmbedder:
    """Deterministic in-memory embedder. Same text → same vector."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.call_log: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.call_log.append(list(texts))
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        return rng.standard_normal(self.dim).tolist()


def make_pipeline(
    tmp_path: Path, *, dedupe_threshold: float = 0.95
) -> tuple[IngestPipeline, FakeEmbedder, ChromaStore, BM25Store]:
    chroma = ChromaStore(tmp_path / "chroma")
    bm25 = BM25Store(tmp_path / "bm25.pkl")
    embedder = FakeEmbedder(dim=16)
    pipeline = IngestPipeline(
        chunkers={"fixed": FixedSizeChunker(size=400, overlap=50), "recursive": RecursiveChunker(size=400, overlap=50)},
        embedder=embedder,
        chroma_store=chroma,
        bm25_store=bm25,
        dedupe_threshold=dedupe_threshold,
    )
    return pipeline, embedder, chroma, bm25


async def test_populates_both_stores(tmp_path: Path) -> None:
    pipeline, _, chroma, bm25 = make_pipeline(tmp_path)
    paths = sorted(FIXTURE_DIR.iterdir())
    result = await pipeline.run(paths)

    assert isinstance(result, IngestResult)
    assert result.docs_loaded == len(paths)
    assert result.chunks_inserted > 0
    assert chroma.count() == bm25.count() == result.chunks_inserted


async def test_counts_match_after_run(tmp_path: Path) -> None:
    pipeline, _, chroma, bm25 = make_pipeline(tmp_path)
    await pipeline.run(sorted(FIXTURE_DIR.iterdir()))
    assert chroma.count() == bm25.count()


async def test_running_twice_does_not_double_insert(tmp_path: Path) -> None:
    pipeline, _, chroma, bm25 = make_pipeline(tmp_path)
    paths = sorted(FIXTURE_DIR.iterdir())
    first = await pipeline.run(paths)
    second = await pipeline.run(paths)

    assert chroma.count() == first.chunks_inserted
    assert bm25.count() == first.chunks_inserted
    assert second.chunks_inserted == 0
    assert second.chunks_skipped_duplicate > 0


async def test_dedup_catches_near_duplicate_across_docs(tmp_path: Path) -> None:
    doc1 = tmp_path / "a.md"
    doc2 = tmp_path / "b.md"
    body = "This is a unique paragraph appearing in two different documents."
    doc1.write_text(body)
    doc2.write_text(body)

    pipeline, _, chroma, bm25 = make_pipeline(tmp_path)
    result = await pipeline.run([doc1, doc2])

    assert result.chunks_produced >= 2
    assert result.chunks_skipped_duplicate >= 1
    assert chroma.count() == bm25.count()


async def test_empty_paths_no_op(tmp_path: Path) -> None:
    pipeline, _, chroma, bm25 = make_pipeline(tmp_path)
    result = await pipeline.run([])
    assert result.chunks_produced == 0
    assert chroma.count() == 0
    assert bm25.count() == 0


async def test_recursive_chunker_works(tmp_path: Path) -> None:
    pipeline, _, chroma, bm25 = make_pipeline(tmp_path)
    result = await pipeline.run([FIXTURE_DIR / "intro.md"], chunker_name="recursive")
    assert result.chunks_inserted > 0
    assert chroma.count() == bm25.count()


async def test_unknown_chunker_raises(tmp_path: Path) -> None:
    pipeline, _, _, _ = make_pipeline(tmp_path)
    with pytest.raises(ValueError, match="Unknown chunker"):
        await pipeline.run([FIXTURE_DIR / "intro.md"], chunker_name="semantic")


async def test_embedder_called_with_chunk_texts(tmp_path: Path) -> None:
    pipeline, embedder, _, _ = make_pipeline(tmp_path)
    await pipeline.run([FIXTURE_DIR / "notes.txt"])
    # All call args concatenated should contain content from the fixture
    all_texts = [t for batch in embedder.call_log for t in batch]
    assert any("kubectl" in t for t in all_texts)


async def test_result_skipped_includes_similarities(tmp_path: Path) -> None:
    doc1 = tmp_path / "a.md"
    doc2 = tmp_path / "b.md"
    body = "Identical content for dedup verification."
    doc1.write_text(body)
    doc2.write_text(body)

    pipeline, _, _, _ = make_pipeline(tmp_path)
    result = await pipeline.run([doc1, doc2])

    assert result.skipped
    assert all(s.similarity >= 0.95 for s in result.skipped)
    assert all(s.chunk_id for s in result.skipped)


async def test_dedup_threshold_respected(tmp_path: Path) -> None:
    doc1 = tmp_path / "a.md"
    doc2 = tmp_path / "b.md"
    body = "Identical content."
    doc1.write_text(body)
    doc2.write_text(body)

    # Threshold 1.01 means nothing will ever be deduped
    pipeline, _, chroma, bm25 = make_pipeline(tmp_path, dedupe_threshold=1.01)
    result = await pipeline.run([doc1, doc2])
    assert result.chunks_skipped_duplicate == 0
    assert chroma.count() == bm25.count() == result.chunks_inserted


async def test_integrity_check_passes_after_run(tmp_path: Path) -> None:
    pipeline, _, _, _ = make_pipeline(tmp_path)
    # Should not raise
    await pipeline.run(sorted(FIXTURE_DIR.iterdir()))


async def test_integrity_error_class_is_exported(tmp_path: Path) -> None:
    assert issubclass(IngestIntegrityError, RuntimeError)
