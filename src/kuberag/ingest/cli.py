import argparse
import asyncio
import logging
import sys
from pathlib import Path

from openai import AsyncOpenAI

from kuberag.config import Settings
from kuberag.ingest.chunkers import Chunker, FixedSizeChunker, RecursiveChunker
from kuberag.ingest.embedder import Embedder, EmbeddingCache
from kuberag.ingest.k8s_source import K8sDocsSource
from kuberag.ingest.pipeline import IngestPipeline, IngestResult
from kuberag.stores import BM25Store, ChromaStore

_SUPPORTED_EXTENSIONS = (".md", ".markdown", ".html", ".htm", ".txt", ".pdf")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kuberag.ingest",
        description="Index documents into the dense and sparse stores.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--path",
        type=Path,
        help="File or directory of documents to ingest.",
    )
    target.add_argument(
        "--source",
        choices=["k8s"],
        help="Pull from a built-in corpus.",
    )
    parser.add_argument(
        "--chunker",
        choices=["fixed", "recursive"],
        default="fixed",
        help="Chunking strategy to use.",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Git commit SHA to pin (only used with --source k8s).",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace, settings: Settings) -> list[Path]:
    if args.source == "k8s":
        source = K8sDocsSource(commit=args.commit)
        return source.fetch(settings.raw_docs_path)

    target = Path(args.path)
    if not target.exists():
        raise FileNotFoundError(target)
    if target.is_file():
        return [target]
    paths: list[Path] = []
    for ext in _SUPPORTED_EXTENSIONS:
        paths.extend(target.rglob(f"*{ext}"))
    return sorted(paths)


def build_pipeline(settings: Settings) -> IngestPipeline:
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    embedder = Embedder(
        model=settings.embedding_model,
        client=client,
        cache=EmbeddingCache(settings.chroma_path.parent / "embedding_cache"),
    )
    chunkers: dict[str, Chunker] = {
        "fixed": FixedSizeChunker(
            size=settings.chunk_size, overlap=settings.chunk_overlap
        ),
        "recursive": RecursiveChunker(
            size=settings.chunk_size, overlap=settings.chunk_overlap
        ),
    }
    return IngestPipeline(
        chunkers=chunkers,
        embedder=embedder,
        chroma_store=ChromaStore(settings.chroma_path),
        bm25_store=BM25Store(settings.bm25_path),
        dedupe_threshold=settings.dedupe_threshold,
    )


def _report(result: IngestResult) -> None:
    print(f"[ingest] docs_loaded={result.docs_loaded}")
    print(f"[ingest] chunks_produced={result.chunks_produced}")
    print(f"[ingest] chunks_inserted={result.chunks_inserted}")
    print(f"[ingest] chunks_skipped_duplicate={result.chunks_skipped_duplicate}")
    print(f"[ingest] chroma_count={result.chroma_count}")
    print(f"[ingest] bm25_count={result.bm25_count}")


async def _run(args: argparse.Namespace, settings: Settings) -> IngestResult:
    paths = resolve_paths(args, settings)
    pipeline = build_pipeline(settings)
    return await pipeline.run(paths, chunker_name=args.chunker)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    settings = Settings()  # type: ignore[call-arg]  # reads required key from env
    result = asyncio.run(_run(args, settings))
    _report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
