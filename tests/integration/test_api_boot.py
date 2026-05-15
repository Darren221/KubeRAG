import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kuberag.api import (
    create_app,
    get_bm25_store,
    get_chroma_store,
    get_hybrid_search,
    get_ingest_pipeline,
    get_orchestrator,
    get_settings,
)
from kuberag.config import Settings
from kuberag.generation.orchestrator import GenerationOrchestrator
from kuberag.ingest.pipeline import IngestPipeline
from kuberag.retrieval.hybrid import HybridSearch
from kuberag.stores import BM25Store, ChromaStore

pytestmark = pytest.mark.integration


def test_create_app_returns_fastapi(test_settings: Settings) -> None:
    app = create_app(test_settings)
    assert isinstance(app, FastAPI)


def test_settings_attached_to_app_state(test_settings: Settings) -> None:
    app = create_app(test_settings)
    assert app.state.settings is test_settings


def test_app_has_title_and_version(test_settings: Settings) -> None:
    app = create_app(test_settings)
    assert app.title == "KubeRAG"
    assert app.version


def test_lifespan_constructs_components(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app):
        assert isinstance(app.state.chroma, ChromaStore)
        assert isinstance(app.state.bm25, BM25Store)
        assert isinstance(app.state.hybrid_search, HybridSearch)
        assert isinstance(app.state.orchestrator, GenerationOrchestrator)
        assert isinstance(app.state.pipeline, IngestPipeline)


def test_chroma_store_uses_configured_path(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app):
        assert app.state.chroma.path == test_settings.chroma_path


def test_dependency_provider_returns_state(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app):
        from starlette.requests import Request

        scope = {"type": "http", "app": app, "headers": []}
        request = Request(scope)  # type: ignore[arg-type]
        assert get_settings(request) is test_settings
        assert get_chroma_store(request) is app.state.chroma
        assert get_bm25_store(request) is app.state.bm25
        assert get_hybrid_search(request) is app.state.hybrid_search
        assert get_orchestrator(request) is app.state.orchestrator
        assert get_ingest_pipeline(request) is app.state.pipeline
