from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI
from starlette.requests import Request

from kuberag import __version__
from kuberag.config import Settings
from kuberag.generation.citations import CitationVerifier
from kuberag.generation.generator import Generator
from kuberag.generation.orchestrator import CompletenessJudge, GenerationOrchestrator
from kuberag.ingest.chunkers import Chunker, FixedSizeChunker, RecursiveChunker
from kuberag.ingest.embedder import Embedder, EmbeddingCache
from kuberag.ingest.pipeline import IngestPipeline
from kuberag.retrieval.dense import DenseRetriever
from kuberag.retrieval.hybrid import HybridSearch
from kuberag.retrieval.reranker import Reranker
from kuberag.retrieval.sparse import SparseRetriever
from kuberag.stores import BM25Store, ChromaStore


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    chroma = ChromaStore(settings.chroma_path)
    bm25 = BM25Store(settings.bm25_path)

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

    pipeline = IngestPipeline(
        chunkers=chunkers,
        embedder=embedder,
        chroma_store=chroma,
        bm25_store=bm25,
        dedupe_threshold=settings.dedupe_threshold,
    )

    hybrid = HybridSearch(
        dense=DenseRetriever(embedder=embedder, store=chroma),
        sparse=SparseRetriever(store=bm25),
        reranker=Reranker(
            client=client, model=settings.judge_model, top_n=settings.rerank_top_n
        ),
        dense_weight=settings.rrf_dense_weight,
    )

    orchestrator = GenerationOrchestrator(
        generator=Generator(client=client, model=settings.generation_model),
        verifier=CitationVerifier(client=client, model=settings.judge_model),
        completeness_judge=CompletenessJudge(
            client=client, model=settings.judge_model
        ),
        confidence_threshold=settings.confidence_threshold,
    )

    app.state.client = client
    app.state.chroma = chroma
    app.state.bm25 = bm25
    app.state.embedder = embedder
    app.state.pipeline = pipeline
    app.state.hybrid_search = hybrid
    app.state.orchestrator = orchestrator

    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]
    app = FastAPI(
        title="KubeRAG",
        version=__version__,
        description="Hybrid-search RAG over Kubernetes documentation.",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    return app


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_chroma_store(request: Request) -> ChromaStore:
    return request.app.state.chroma  # type: ignore[no-any-return]


def get_bm25_store(request: Request) -> BM25Store:
    return request.app.state.bm25  # type: ignore[no-any-return]


def get_hybrid_search(request: Request) -> HybridSearch:
    return request.app.state.hybrid_search  # type: ignore[no-any-return]


def get_orchestrator(request: Request) -> GenerationOrchestrator:
    return request.app.state.orchestrator  # type: ignore[no-any-return]


def get_ingest_pipeline(request: Request) -> IngestPipeline:
    return request.app.state.pipeline  # type: ignore[no-any-return]
