import hashlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kuberag.api import create_app, get_ingest_pipeline
from kuberag.config import Settings
from kuberag.ingest.chunkers import FixedSizeChunker, RecursiveChunker
from kuberag.ingest.pipeline import IngestPipeline

pytestmark = pytest.mark.integration

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "docs"


class FakeEmbedder:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        return rng.standard_normal(self.dim).tolist()


def _override_pipeline_with_fake_embedder(app: object) -> IngestPipeline:
    """After lifespan has run, swap in an IngestPipeline that uses a fake embedder
    but the lifespan-created Chroma and BM25 stores."""
    state = app.state  # type: ignore[attr-defined]
    fake_pipeline = IngestPipeline(
        chunkers={
            "fixed": FixedSizeChunker(size=400, overlap=50),
            "recursive": RecursiveChunker(size=400, overlap=50),
        },
        embedder=FakeEmbedder(),
        chroma_store=state.chroma,
        bm25_store=state.bm25,
    )
    app.dependency_overrides[get_ingest_pipeline] = lambda: fake_pipeline  # type: ignore[attr-defined]
    return fake_pipeline


def test_ingest_path_round_trip(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        _override_pipeline_with_fake_embedder(app)
        response = client.post(
            "/v1/ingest", json={"path": str(FIXTURE_DIR)}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["docs_loaded"] >= 1
        assert body["chunks_inserted"] > 0
        assert body["chroma_count"] == body["bm25_count"]


def test_chroma_count_increases_after_ingest(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        _override_pipeline_with_fake_embedder(app)
        before = app.state.chroma.count()
        client.post("/v1/ingest", json={"path": str(FIXTURE_DIR / "intro.md")})
        after = app.state.chroma.count()
        assert after > before


def test_rerun_reports_skipped_duplicates(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        _override_pipeline_with_fake_embedder(app)
        first = client.post("/v1/ingest", json={"path": str(FIXTURE_DIR)}).json()
        second = client.post("/v1/ingest", json={"path": str(FIXTURE_DIR)}).json()
        assert first["chunks_inserted"] > 0
        assert second["chunks_inserted"] == 0
        assert second["chunks_skipped_duplicate"] > 0


def test_chunker_choice_recursive(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        _override_pipeline_with_fake_embedder(app)
        body = client.post(
            "/v1/ingest",
            json={"path": str(FIXTURE_DIR / "intro.md"), "chunker": "recursive"},
        ).json()
        assert body["chunks_inserted"] > 0


def test_default_chunker_is_fixed(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        _override_pipeline_with_fake_embedder(app)
        response = client.post(
            "/v1/ingest", json={"path": str(FIXTURE_DIR / "notes.txt")}
        )
        assert response.status_code == 200
        docs = client.get("/v1/documents").json()
        # All entries from this ingest should be 'fixed'
        for entry in docs:
            assert entry["chunking_strategy"] == "fixed"


def test_invalid_path_returns_error(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        _override_pipeline_with_fake_embedder(app)
        response = client.post(
            "/v1/ingest", json={"path": "/nonexistent/path/here"}
        )
        assert response.status_code >= 400


def test_invalid_chunker_returns_422(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        _override_pipeline_with_fake_embedder(app)
        response = client.post(
            "/v1/ingest", json={"path": str(FIXTURE_DIR), "chunker": "semantic"}
        )
        assert response.status_code == 422


def test_missing_path_returns_422(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        _override_pipeline_with_fake_embedder(app)
        response = client.post("/v1/ingest", json={})
        assert response.status_code == 422


def test_route_in_openapi(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        spec = client.get("/openapi.json").json()
        assert "/v1/ingest" in spec["paths"]
        assert "post" in spec["paths"]["/v1/ingest"]


def test_response_shape(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        _override_pipeline_with_fake_embedder(app)
        body = client.post("/v1/ingest", json={"path": str(FIXTURE_DIR)}).json()
        required_fields = {
            "docs_loaded",
            "chunks_produced",
            "chunks_inserted",
            "chunks_skipped_duplicate",
            "chroma_count",
            "bm25_count",
        }
        assert required_fields <= set(body.keys())
