import pytest
from fastapi.testclient import TestClient

from kuberag.api import create_app
from kuberag.config import Settings
from kuberag.ingest.chunkers import Chunk

pytestmark = pytest.mark.integration


def make_chunk(
    chunk_id: str,
    *,
    source: str,
    strategy: str = "fixed",
    text: str = "sample",
    index: int = 0,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source=source,
        section=None,
        chunk_index=index,
        chunking_strategy=strategy,
        char_count=len(text),
    )


def test_empty_store_returns_empty_list(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        response = client.get("/v1/documents")
        assert response.status_code == 200
        assert response.json() == []


def test_lists_indexed_documents(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        chunks = [
            make_chunk("a1", source="/docs/intro.md", index=0),
            make_chunk("a2", source="/docs/intro.md", index=1),
            make_chunk("b1", source="/docs/networking.md", index=0),
        ]
        embeddings = [[float(i)] * 8 for i in range(len(chunks))]
        app.state.chroma.add(chunks, embeddings)

        body = client.get("/v1/documents").json()
        sources = {entry["source"] for entry in body}
        assert sources == {"/docs/intro.md", "/docs/networking.md"}


def test_chunk_counts_correct(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        chunks = [
            make_chunk("a1", source="/docs/intro.md", index=0),
            make_chunk("a2", source="/docs/intro.md", index=1),
            make_chunk("a3", source="/docs/intro.md", index=2),
            make_chunk("b1", source="/docs/networking.md", index=0),
        ]
        app.state.chroma.add(chunks, [[float(i)] * 8 for i in range(len(chunks))])

        body = client.get("/v1/documents").json()
        by_source = {entry["source"]: entry["chunk_count"] for entry in body}
        assert by_source["/docs/intro.md"] == 3
        assert by_source["/docs/networking.md"] == 1


def test_separates_by_chunking_strategy(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        chunks = [
            make_chunk("a1", source="/docs/intro.md", strategy="fixed", index=0),
            make_chunk("a2", source="/docs/intro.md", strategy="recursive", index=0),
        ]
        app.state.chroma.add(chunks, [[float(i)] * 8 for i in range(2)])

        body = client.get("/v1/documents").json()
        assert len(body) == 2
        strategies = {entry["chunking_strategy"] for entry in body}
        assert strategies == {"fixed", "recursive"}


def test_response_is_sorted(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        chunks = [
            make_chunk("c1", source="/docs/charlie.md"),
            make_chunk("a1", source="/docs/alpha.md"),
            make_chunk("b1", source="/docs/bravo.md"),
        ]
        app.state.chroma.add(chunks, [[float(i)] * 8 for i in range(3)])

        body = client.get("/v1/documents").json()
        sources = [entry["source"] for entry in body]
        assert sources == sorted(sources)


def test_route_in_openapi(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        spec = client.get("/openapi.json").json()
        assert "/v1/documents" in spec["paths"]


def test_response_entry_shape(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        chunk = make_chunk("a1", source="/docs/intro.md")
        app.state.chroma.add([chunk], [[0.1] * 8])
        body = client.get("/v1/documents").json()
        entry = body[0]
        assert set(entry.keys()) == {"source", "chunking_strategy", "chunk_count"}
