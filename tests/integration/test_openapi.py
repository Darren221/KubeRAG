import pytest
from fastapi.testclient import TestClient

from kuberag.api import create_app
from kuberag.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture
def spec(test_settings: Settings) -> dict:  # type: ignore[type-arg]
    app = create_app(test_settings)
    with TestClient(app) as client:
        return client.get("/openapi.json").json()  # type: ignore[no-any-return]


def test_top_level_description_present(spec: dict) -> None:  # type: ignore[type-arg]
    assert "Hybrid-search" in spec["info"]["description"]


def test_tag_groups_declared(spec: dict) -> None:  # type: ignore[type-arg]
    names = {tag["name"] for tag in spec.get("tags", [])}
    assert {"retrieval", "ingestion", "health"} <= names


def test_each_route_has_description_and_operation_id(spec: dict) -> None:  # type: ignore[type-arg]
    expected = {
        ("/v1/health", "get"): "health",
        ("/v1/ask", "post"): "ask",
        ("/v1/documents", "get"): "list_documents",
        ("/v1/ingest", "post"): "ingest",
    }
    for (path, method), op_id in expected.items():
        operation = spec["paths"][path][method]
        assert operation["operationId"] == op_id
        assert operation.get("description")
        assert operation.get("summary")


def test_each_route_tagged(spec: dict) -> None:  # type: ignore[type-arg]
    tag_by_route = {
        ("/v1/health", "get"): "health",
        ("/v1/ask", "post"): "retrieval",
        ("/v1/documents", "get"): "retrieval",
        ("/v1/ingest", "post"): "ingestion",
    }
    for (path, method), tag in tag_by_route.items():
        assert tag in spec["paths"][path][method]["tags"]


def test_ask_request_example_present(spec: dict) -> None:  # type: ignore[type-arg]
    schema = spec["components"]["schemas"]["AskRequest"]
    examples = schema.get("examples")
    assert examples and isinstance(examples, list)
    assert "question" in examples[0]


def test_ingest_request_example_present(spec: dict) -> None:  # type: ignore[type-arg]
    schema = spec["components"]["schemas"]["IngestRequest"]
    examples = schema.get("examples")
    assert examples and isinstance(examples, list)
    assert "path" in examples[0]


def test_ingest_documents_404_response_documented(spec: dict) -> None:  # type: ignore[type-arg]
    responses = spec["paths"]["/v1/ingest"]["post"]["responses"]
    assert "404" in responses
