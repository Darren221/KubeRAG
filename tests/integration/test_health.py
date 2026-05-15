import pytest
from fastapi.testclient import TestClient

from kuberag.api import create_app
from kuberag.config import Settings
from kuberag.ingest.chunkers import Chunk

pytestmark = pytest.mark.integration


def make_chunk(chunk_id: str, text: str = "sample") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source=f"/test/{chunk_id}.md",
        section=None,
        chunk_index=0,
        chunking_strategy="fixed",
        char_count=len(text),
    )


def test_returns_200(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        response = client.get("/v1/health")
        assert response.status_code == 200


def test_response_has_expected_shape(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        response = client.get("/v1/health")
        body = response.json()
        assert body["status"] == "ok"
        assert isinstance(body["chroma_count"], int)
        assert isinstance(body["bm25_count"], int)
        assert "models" in body


def test_reports_configured_models(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        response = client.get("/v1/health")
        models = response.json()["models"]
        assert models["embedding"] == test_settings.embedding_model
        assert models["generation"] == test_settings.generation_model
        assert models["judge"] == test_settings.judge_model


def test_counts_are_zero_when_stores_empty(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        body = client.get("/v1/health").json()
        assert body["chroma_count"] == 0
        assert body["bm25_count"] == 0


def test_counts_reflect_populated_stores(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        chunks = [make_chunk(f"c{i}") for i in range(3)]
        embeddings = [[float(i)] * 16 for i in range(3)]
        app.state.chroma.add(chunks, embeddings)
        app.state.bm25.add(chunks)

        body = client.get("/v1/health").json()
        assert body["chroma_count"] == 3
        assert body["bm25_count"] == 3


def test_health_listed_in_openapi(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        spec = client.get("/openapi.json").json()
        assert "/v1/health" in spec["paths"]
